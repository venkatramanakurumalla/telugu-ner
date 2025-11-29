import re
import os
import pickle
import json
from collections import Counter
import sys
from typing import List, Tuple, Dict, Any, Optional

# Try to import ML dependencies, gracefully fail if missing
try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics
    CRF_AVAILABLE = True
except ImportError:
    print("Warning: 'sklearn-crfsuite' not found. CRF module will be disabled.")
    sklearn_crfsuite = None
    CRF_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: 'numpy' not found. Some features disabled.")
    NUMPY_AVAILABLE = False

# ==========================================
# MODULE A: ADVANCED SUFFIX STRIPPER
# ==========================================
class AdvancedSuffixStripper:
    def __init__(self):
        # Comprehensive Telugu suffix patterns with weights and contexts
        self.suffix_patterns = [
            # Complex Postpositions (High priority)
            (r'కోసం$', '', 1.0, 'POSTPOSITION'),
            (r'నుండి$', '', 1.0, 'POSTPOSITION'), 
            (r'కాబట్టి$', '', 1.0, 'POSTPOSITION'),
            (r'వరకు$', '', 1.0, 'POSTPOSITION'),
            (r'ద్వారా$', '', 1.0, 'POSTPOSITION'),
            (r'వల్ల$', '', 1.0, 'POSTPOSITION'),
            
            # Location Markers (Medium priority)
            (r'లోని$', '', 0.9, 'LOCATIVE'),
            (r'లో$', '', 0.9, 'LOCATIVE'),
            (r'మీద$', '', 0.9, 'LOCATIVE'),
            (r'పై$', '', 0.9, 'LOCATIVE'),
            (r'కడ$', '', 0.9, 'LOCATIVE'),
            (r'వద్ద$', '', 0.9, 'LOCATIVE'),
            
            # Dative/Case Markers (Medium priority)
            (r'కు$', '', 0.8, 'DATIVE'),
            (r'కి$', '', 0.8, 'DATIVE'),
            (r'తో$', '', 0.8, 'INSTRUMENTAL'),
            (r'తోటి$', '', 0.8, 'INSTRUMENTAL'),
            
            # Accusative Markers
            (r'ని$', '', 0.7, 'ACCUSATIVE'),
            (r'ను$', '', 0.7, 'ACCUSATIVE'),
            
            # Plural Markers (Low priority)
            (r'లు$', '', 0.6, 'PLURAL'),
            (r'ల$', '', 0.6, 'PLURAL'),
            (r'ళ్లు$', '', 0.6, 'PLURAL'),
            (r'ళ్ల$', '', 0.6, 'PLURAL'),
            (r'రు$', '', 0.6, 'PLURAL'),  # For persons
        ]
        
        # Advanced Sandhi transformation rules
        self.sandhi_transforms = [
            # Masculine noun restorations
            (r'డి$', 'డు', 0.9, 'MASCULINE'),
            (r'ని$', 'డు', 0.8, 'MASCULINE'),
            (r'టి$', 'టు', 0.9, 'MASCULINE'),
            (r'రి$', 'రు', 0.8, 'MASCULINE'),
            
            # Neuter/Feminine restorations  
            (r'తి$', 'తు', 0.7, 'NEUTER'),
            (r'ది$', 'దు', 0.7, 'NEUTER'),
            
            # Vowel harmony fixes
            (r'య్య$', 'య', 0.6, 'VOWEL_HARMONY'),
            (r'వ్వ$', 'వ', 0.6, 'VOWEL_HARMONY'),
        ]
        
        # Protected words that shouldn't be stripped
        self.protected_words = {
            'లో', 'మీద', 'పై', 'కు', 'కి', 'తో', 'ని', 'ను',
            'అతను', 'ఆమె', 'నేను', 'మేము', 'తను', 'వారు'
        }
        
        # Minimum root length by original word length
        self.min_root_rules = {
            2: 1,  # 2-char words can reduce to 1
            3: 2,  # 3-char words can reduce to 2  
            4: 2,  # 4-char words can reduce to 2
            5: 3,  # 5-char words can reduce to 3
        }

    def get_min_root_length(self, original_word: str) -> int:
        """Calculate minimum allowed root length based on original word"""
        length = len(original_word)
        for threshold, min_len in sorted(self.min_root_rules.items()):
            if length <= threshold:
                return min_len
        return 3  # Default for longer words

    def get_root(self, word: str) -> Tuple[str, List[Dict]]:
        """Get root word with transformation history"""
        original_word = word.strip()
        if not original_word or original_word in self.protected_words:
            return original_word, []
            
        transformation_log = []
        current_word = original_word
        min_root_length = self.get_min_root_length(original_word)
        
        # Phase 1: Suffix Stripping with confidence scoring
        for pattern, replacement, weight, category in self.suffix_patterns:
            if re.search(pattern, current_word):
                candidate = re.sub(pattern, replacement, current_word)
                if len(candidate) >= min_root_length:
                    transformation_log.append({
                        'from': current_word,
                        'to': candidate, 
                        'type': 'SUFFIX_STRIP',
                        'pattern': pattern,
                        'category': category,
                        'confidence': weight
                    })
                    current_word = candidate
                    break  # One suffix at a time for now
        
        # Phase 2: Sandhi Transformations  
        for pattern, replacement, weight, category in self.sandhi_transforms:
            if re.search(pattern, current_word):
                candidate = re.sub(pattern, replacement, current_word)
                transformation_log.append({
                    'from': current_word,
                    'to': candidate,
                    'type': 'SANDHI_FIX', 
                    'pattern': pattern,
                    'category': category,
                    'confidence': weight
                })
                current_word = candidate
                break
        
        return current_word, transformation_log

    def analyze_suffixes(self, word: str) -> Dict[str, Any]:
        """Comprehensive suffix analysis"""
        root, transformations = self.get_root(word)
        
        detected_suffixes = []
        for trans in transformations:
            if trans['type'] == 'SUFFIX_STRIP':
                detected_suffixes.append({
                    'suffix': trans['pattern'].replace('$', '').replace('^', ''),
                    'category': trans['category'],
                    'confidence': trans['confidence']
                })
        
        return {
            'original': word,
            'root': root,
            'transformations': transformations,
            'detected_suffixes': detected_suffixes,
            'root_length_ratio': len(root) / len(word) if word else 0
        }

# ==========================================
# MODULE B: INTELLIGENT RULE ENGINE
# ==========================================
class IntelligentRuleEngine:
    def __init__(self):
        self.stripper = AdvancedSuffixStripper()
        
        # Enhanced suffix databases with confidence scores
        self.location_indicators = [
            ('బాద్', 0.9), ('పురం', 0.95), ('పట్నం', 0.9), ('నగర్', 0.85),
            ('పల్లి', 0.8), ('పేట', 0.8), ('ఊరు', 0.9), ('వాడ', 0.7),
            ('గిరి', 0.8), ('కొండ', 0.8), ('గూడెం', 0.7), ('చెరువు', 0.6)
        ]
        
        self.male_indicators = [
            ('డు', 0.8), ('య్య', 0.7), ('రావు', 0.9), ('రెడ్డి', 0.85),
            ('బాబు', 0.8), ('శాస్త్రి', 0.9), ('రాజు', 0.8), ('రావు', 0.85),
            ('నాథ్', 0.7), ('గౌడ్', 0.8), ('చారి', 0.7)
        ]
        
        self.female_indicators = [
            ('మ్మ', 0.8), ('దేవి', 0.9), ('లత', 0.7), ('శ్రీ', 0.6),
            ('రాణి', 0.8), ('వతి', 0.7), ('సీ', 0.6), ('బాయి', 0.7)
        ]
        
        self.object_indicators = [
            ('ము', 0.7), ('ణం', 0.8), ('కం', 0.7), ('పు', 0.6),
            ('వు', 0.6), ('తు', 0.5), ('లు', 0.5), ('డు', 0.4)
        ]
        
        # Context patterns for improved disambiguation
        self.context_patterns = {
            'LOCATION': [r'.*పట్నం', r'.*నగర', r'.*గ్రామం', r'.*స్టేషన్'],
            'PERSON': [r'.*రావు', r'.*రెడ్డి', r'.*శర్మ', r'.*కుమార్'],
            'OBJECT': [r'.*పుస్తక', r'.*యంత్ర', r'.*సామగ్రి']
        }

    def calculate_pattern_confidence(self, word: str, patterns: List[Tuple[str, float]]) -> float:
        """Calculate confidence based on suffix matching"""
        max_confidence = 0.0
        for suffix, confidence in patterns:
            if word.endswith(suffix):
                # Longer suffixes get bonus
                length_bonus = min(0.2, len(suffix) * 0.05)
                max_confidence = max(max_confidence, confidence + length_bonus)
        return max_confidence

    def predict_with_confidence(self, raw_word: str) -> List[Tuple[str, float, str]]:
        """Predict with confidence scores for all categories"""
        root_analysis = self.stripper.analyze_suffixes(raw_word)
        root = root_analysis['root']
        
        predictions = []
        
        # Location detection
        loc_conf = self.calculate_pattern_confidence(root, self.location_indicators)
        if loc_conf > 0:
            predictions.append(("NOUN_LOC", loc_conf, "SUFFIX_PATTERN"))
        
        # Person detection  
        male_conf = self.calculate_pattern_confidence(root, self.male_indicators)
        female_conf = self.calculate_pattern_confidence(root, self.female_indicators)
        person_conf = max(male_conf, female_conf)
        if person_conf > 0:
            predictions.append(("NOUN_PER", person_conf, "PERSON_SUFFIX"))
        
        # Object detection
        obj_conf = self.calculate_pattern_confidence(root, self.object_indicators)
        if obj_conf > 0:
            predictions.append(("NOUN_OBJ", obj_conf, "OBJECT_SUFFIX"))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

    def get_detailed_analysis(self, raw_word: str) -> Dict[str, Any]:
        """Comprehensive rule-based analysis"""
        root_analysis = self.stripper.analyze_suffixes(raw_word)
        predictions = self.predict_with_confidence(raw_word)
        
        return {
            'word': raw_word,
            'root_analysis': root_analysis,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else ("UNKNOWN", 0.0, "NO_RULE"),
            'rule_confidence': predictions[0][1] if predictions else 0.0
        }

# ==========================================
# MODULE C: SMART GAZETTEER
# ==========================================
class SmartGazetteer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.locations = set()
        self.persons = set()
        self.objects = set()
        self.pronouns = set()
        
        # Frequency data for confidence calculation
        self.word_frequencies = Counter()
        
        self.load_base_vocabulary()
        self.load_external_data()
        
    def load_base_vocabulary(self):
        """Load core Telugu vocabulary"""
        # Locations
        self.locations = {
            "హైదరాబాద్", "వరంగల్", "విజయవాడ", "బెంగళూరు", "చెన్నై",
            "దిల్లీ", "కేరళ", "ముంబై", "కలకత్తా", "అమరావతి",
            "సంత", "ఇల్లు", "బడి", "ఆస్పత్రి", "యూనివర్సిటీ"
        }
        
        # Persons and Pronouns
        self.persons = {
            "రాము", "కృష్ణ", "సీత", "రావణ", "అర్జున",
            "కిరణ్", "ఆనంద్", "రాజు", "పవన్", "సుమన్",
            "గీత", "ప్రియ", "మోహన్", "సుధా", "లక్ష్మి"
        }
        
        self.pronouns = {
            "అతను", "ఆమె", "నేను", "మేము", "తను", "వారు", "అది",
            "ఇది", "అవి", "ఇవి", "నీవు", "మీరు", "తమరు"
        }
        
        # Objects
        self.objects = {
            "పుస్తకం", "మేజా", "పెన్ను", "కారు", "బస్సు", "చీర",
            "టీ", "పండ్లు", "అన్నం", "ఉత్తరం", "బహుమతి", "గంట",
            "చెవి", "కాలు", "చేయి", "తల", "బట్ట", "ఇల్లు"
        }
        
        # Initialize frequencies
        for word in list(self.locations) + list(self.persons) + list(self.objects) + list(self.pronouns):
            self.word_frequencies[word] = 100  # Base frequency

    def load_external_data(self):
        """Load additional data from files if available"""
        try:
            # Load location data
            loc_file = os.path.join(self.data_dir, "locations.txt")
            if os.path.exists(loc_file):
                with open(loc_file, 'r', encoding='utf-8') as f:
                    self.locations.update(line.strip() for line in f)
                    
            # Load person names
            person_file = os.path.join(self.data_dir, "persons.txt") 
            if os.path.exists(person_file):
                with open(person_file, 'r', encoding='utf-8') as f:
                    self.persons.update(line.strip() for line in f)
                    
        except Exception as e:
            print(f"Warning: Could not load external data: {e}")

    def predict_with_confidence(self, root_word: str) -> List[Tuple[str, float]]:
        """Predict with confidence scores"""
        predictions = []
        
        # Exact match with high confidence
        if root_word in self.pronouns:
            predictions.append(("NOUN_PER", 0.99))
        elif root_word in self.locations:
            predictions.append(("NOUN_LOC", 0.95))
        elif root_word in self.persons:
            predictions.append(("NOUN_PER", 0.92))
        elif root_word in self.objects:
            predictions.append(("NOUN_OBJ", 0.90))
        
        # Partial match with lower confidence
        if not predictions:
            for loc in self.locations:
                if root_word in loc or loc in root_word:
                    predictions.append(("NOUN_LOC", 0.7))
                    break
                    
            for person in self.persons:
                if root_word in person or person in root_word:
                    predictions.append(("NOUN_PER", 0.7))
                    break
        
        return predictions

    def add_word(self, word: str, category: str, frequency: int = 1):
        """Dynamically add word to gazetteer"""
        if category == "NOUN_LOC":
            self.locations.add(word)
        elif category == "NOUN_PER":
            self.persons.add(word)
        elif category == "NOUN_OBJ":
            self.objects.add(word)
            
        self.word_frequencies[word] += frequency

    def get_statistics(self) -> Dict[str, Any]:
        """Get gazetteer statistics"""
        return {
            'total_locations': len(self.locations),
            'total_persons': len(self.persons),
            'total_objects': len(self.objects),
            'total_pronouns': len(self.pronouns),
            'total_vocabulary': len(self.locations) + len(self.persons) + len(self.objects) + len(self.pronouns)
        }

# ==========================================
# MODULE D: ADVANCED CRF ENGINE
# ==========================================
class AdvancedCRFEngine:
    def __init__(self, model_path: str = "models/crf_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_stats = {}
        
        if CRF_AVAILABLE:
            self.initialize_model()

    def initialize_model(self):
        """Initialize CRF model with optimal parameters"""
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=200,
            all_possible_transitions=True,
            verbose=False
        )

    def extract_features(self, sent: List[str], i: int) -> Dict[str, Any]:
        """Enhanced feature extraction for Telugu"""
        word = sent[i]
        
        features = {
            'bias': 1.0,
            'word.length': len(word),
            'word.lower': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[:3]': word[:3],
            'word[:2]': word[:2],
            'word.isdigit()': word.isdigit(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
        }
        
        # Telugu-specific features
        features.update({
            'has_telugu_vowel': bool(re.search(r'[అ-ఔ]', word)),
            'has_telugu_consonant': bool(re.search(r'[క-హ]', word)),
            'ends_with_vowel': bool(re.search(r'[అ-ఔ]$', word)),
            'starts_with_consonant': bool(re.search(r'^[క-హ]', word)),
        })
        
        # Context features
        for offset in [-2, -1, 1, 2]:
            if 0 <= i + offset < len(sent):
                features.update({
                    f'{offset}:word': sent[i + offset],
                    f'{offset}:word.lower': sent[i + offset].lower(),
                    f'{offset}:word.length': len(sent[i + offset]),
                })
            else:
                features[f'offset_{offset}_BOS/EOS'] = True
        
        return features

    def sent_to_features(self, sent: List[str]) -> List[Dict[str, Any]]:
        return [self.extract_features(sent, i) for i in range(len(sent))]

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the CRF model"""
        if not CRF_AVAILABLE or not self.model:
            print("CRF not available, skipping training")
            return
            
        try:
            print("Training CRF model...")
            self.model.fit(
                [self.sent_to_features(s) for s in X_train],
                y_train
            )
            self.is_trained = True
            print("CRF training completed")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            print(f"CRF training failed: {e}")

    def predict(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Predict with confidence scores"""
        if not self.is_trained or not self.model:
            return [("UNK", 0.0)] * len(tokens)
            
        try:
            features = [self.sent_to_features(tokens)]
            predictions = self.model.predict(features)[0]
            
            # Calculate marginal probabilities for confidence
            confidences = []
            marginals = self.model.predict_marginals(features)[0]
            
            for i, pred in enumerate(predictions):
                confidence = marginals[i].get(pred, 0.0)
                confidences.append((pred, float(confidence)))
                
            return confidences
            
        except Exception as e:
            print(f"CRF prediction error: {e}")
            return [("UNK", 0.0)] * len(tokens)

    def save_model(self):
        """Save trained model to disk"""
        if self.model and self.is_trained:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
                self.is_trained = True
                print("CRF model loaded from disk")

    def train_with_default_data(self):
        """Train with some default Telugu patterns"""
        if not CRF_AVAILABLE:
            return
            
        # Default training data based on common patterns
        X_train = [
            ['నేను', 'పెన్నుతో', 'రాశాను'],
            ['రాము', 'సంతకు', 'వెళ్ళాడు'],
            ['సీత', 'పుస్తకం', 'చదివింది'],
            ['బస్సు', 'స్టేషన్', 'వద్ద', 'ఉంది'],
            ['అతను', 'ఇల్లు', 'కి', 'వెళ్ళాడు']
        ]
        
        y_train = [
            ['NOUN_PER', 'NOUN_OBJ', 'VERB'],
            ['NOUN_PER', 'NOUN_LOC', 'VERB'],
            ['NOUN_PER', 'NOUN_OBJ', 'VERB'],
            ['NOUN_OBJ', 'NOUN_LOC', 'POSTPOSITION', 'VERB'],
            ['NOUN_PER', 'NOUN_LOC', 'POSTPOSITION', 'VERB']
        ]
        
        self.train(X_train, y_train)

# ==========================================
# MODULE E: ENSEMBLE ORCHESTRATOR
# ==========================================
class PowerNounEngine:
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.stripper = AdvancedSuffixStripper()
        self.rules = IntelligentRuleEngine()
        self.gazetteer = SmartGazetteer(data_dir)
        self.crf = AdvancedCRFEngine(os.path.join(model_dir, "crf_model.pkl"))
        
        # Try to load pre-trained CRF model
        self.crf.load_model()
        if not self.crf.is_trained:
            self.crf.train_with_default_data()
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'module_usage': Counter(),
            'confidence_distribution': []
        }
        
        # Configuration
        self.confidence_thresholds = {
            'gazetteer': 0.8,
            'rules': 0.6,
            'crf': 0.5
        }

    def ensemble_predict(self, word: str, context: List[str] = None) -> Dict[str, Any]:
        """Advanced ensemble prediction with multiple strategies"""
        
        # Get predictions from all modules
        rule_analysis = self.rules.get_detailed_analysis(word)
        gazetteer_preds = self.gazetteer.predict_with_confidence(rule_analysis['root_analysis']['root'])
        
        # Get CRF predictions if context available
        crf_preds = []
        if context:
            crf_preds = self.crf.predict(context)
            # Find the prediction for current word
            word_index = context.index(word) if word in context else -1
            if word_index >= 0:
                crf_preds = [crf_preds[word_index]]
        
        # Ensemble voting with confidence
        candidates = []
        
        # Gazetteer predictions (high weight)
        for tag, conf in gazetteer_preds:
            candidates.append({
                'tag': tag,
                'confidence': conf,
                'source': 'GAZETTEER',
                'weight': 1.0
            })
        
        # Rule-based predictions (medium weight)
        for tag, conf, source in rule_analysis['predictions']:
            candidates.append({
                'tag': tag,
                'confidence': conf,
                'source': f'RULE_{source}',
                'weight': 0.8
            })
        
        # CRF predictions (contextual weight)
        for tag, conf in crf_preds:
            if tag.startswith('NOUN'):
                candidates.append({
                    'tag': tag,
                    'confidence': conf,
                    'source': 'CRF',
                    'weight': 0.9
                })
        
        # Weighted voting
        if candidates:
            weighted_scores = {}
            for candidate in candidates:
                score = candidate['confidence'] * candidate['weight']
                if candidate['tag'] not in weighted_scores:
                    weighted_scores[candidate['tag']] = []
                weighted_scores[candidate['tag']].append(score)
            
            # Average scores per tag
            final_scores = {}
            for tag, scores in weighted_scores.items():
                final_scores[tag] = sum(scores) / len(scores)
            
            # Get best prediction
            best_tag = max(final_scores.items(), key=lambda x: x[1])
            
            return {
                'word': word,
                'root': rule_analysis['root_analysis']['root'],
                'final_tag': best_tag[0],
                'final_confidence': best_tag[1],
                'sources_used': list(set(c['source'] for c in candidates)),
                'all_candidates': candidates,
                'rule_analysis': rule_analysis,
                'gazetteer_predictions': gazetteer_preds,
                'crf_predictions': crf_preds
            }
        
        return {
            'word': word,
            'root': rule_analysis['root_analysis']['root'],
            'final_tag': 'UNKNOWN',
            'final_confidence': 0.0,
            'sources_used': [],
            'all_candidates': [],
            'rule_analysis': rule_analysis
        }

    def analyze_sentence(self, sentence: str, detailed: bool = False) -> Dict[str, Any]:
        """Complete sentence analysis"""
        # Enhanced tokenization for Telugu
        tokens = re.findall(r"[\w\u0C00-\u0C7F]+", sentence)
        
        if not tokens:
            return {'error': 'No tokens found'}
        
        results = []
        overall_confidence = 0.0
        
        print(f"\n🔍 Analyzing: {sentence}")
        print("=" * 80)
        print(f"{'TOKEN':<15} | {'ROOT':<12} | {'TAG':<12} | {'CONF':<6} | {'SOURCES'}")
        print("-" * 80)
        
        for i, token in enumerate(tokens):
            # Get context window
            start = max(0, i-2)
            end = min(len(tokens), i+3)
            context = tokens[start:end]
            
            analysis = self.ensemble_predict(token, context)
            results.append(analysis)
            
            # Update performance stats
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['confidence_distribution'].append(analysis['final_confidence'])
            for source in analysis['sources_used']:
                self.performance_stats['module_usage'][source] += 1
            
            # Display results
            sources_str = ", ".join(analysis['sources_used']) if analysis['sources_used'] else "FAILED"
            print(f"{token:<15} | {analysis['root']:<12} | {analysis['final_tag']:<12} | {analysis['final_confidence']:<6.2f} | {sources_str}")
            
            overall_confidence += analysis['final_confidence']
        
        overall_confidence /= len(tokens) if tokens else 1
        
        print("-" * 80)
        print(f"📊 Overall Confidence: {overall_confidence:.2f}")
        
        return {
            'sentence': sentence,
            'tokens': tokens,
            'analysis': results,
            'overall_confidence': overall_confidence,
            'performance_stats': self.get_performance_stats()
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        total = self.performance_stats['total_predictions']
        if total == 0:
            return {'error': 'No predictions made'}
        
        confidences = self.performance_stats['confidence_distribution']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_predictions': total,
            'accuracy': self.performance_stats['correct_predictions'] / total if total > 0 else 0,
            'average_confidence': avg_confidence,
            'module_usage': dict(self.performance_stats['module_usage']),
            'high_confidence_ratio': len([c for c in confidences if c > 0.7]) / len(confidences) if confidences else 0
        }

    def save_knowledge(self):
        """Save all learned knowledge to disk"""
        # Save gazetteer data
        gazetteer_data = {
            'locations': list(self.gazetteer.locations),
            'persons': list(self.gazetteer.persons),
            'objects': list(self.gazetteer.objects),
            'frequencies': dict(self.gazetteer.word_frequencies)
        }
        
        with open(os.path.join(self.data_dir, 'gazetteer.json'), 'w', encoding='utf-8') as f:
            json.dump(gazetteer_data, f, ensure_ascii=False, indent=2)
        
        # Save performance stats
        with open(os.path.join(self.model_dir, 'performance.json'), 'w') as f:
            json.dump(self.performance_stats, f, indent=2)
        
        print("💾 Knowledge saved successfully!")

    def load_knowledge(self):
        """Load learned knowledge from disk"""
        try:
            # Load gazetteer data
            gazetteer_file = os.path.join(self.data_dir, 'gazetteer.json')
            if os.path.exists(gazetteer_file):
                with open(gazetteer_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.gazetteer.locations = set(data['locations'])
                    self.gazetteer.persons = set(data['persons'])
                    self.gazetteer.objects = set(data['objects'])
                    self.gazetteer.word_frequencies = Counter(data['frequencies'])
            
            # Load performance stats
            performance_file = os.path.join(self.model_dir, 'performance.json')
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    self.performance_stats = json.load(f)
                    
            print("📚 Knowledge loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load saved knowledge: {e}")

# ==========================================
# USAGE AND TESTING
# ==========================================
def main():
    """Comprehensive testing of the PowerNounEngine"""
    
    print("🚀 INITIALIZING POWER NOUN ENGINE FOR TELUGU")
    print("=" * 60)
    
    # Initialize engine
    engine = PowerNounEngine()
    
    # Load any saved knowledge
    engine.load_knowledge()
    
    # Test sentences covering various Telugu noun patterns
    test_sentences = [
        "నేను పెన్నుతో రాశాను",                    # With pen (instrumental)
        "రాముడు సంతకు వెళ్ళాడు",                   # To market (dative)  
        "సీత పుస్తకం చదివింది",                    # Book (object)
        "బస్సు స్టేషన్ వద్ద ఉంది",                 # At station (locative)
        "అతను హైదరాబాద్ నుండి వచ్చాడు",           # From Hyderabad (ablative)
        "పండ్లు బ్యాగులో ఉన్నాయి",                 # Fruits in bag (locative)
        "కిరణ్ రెడ్డి గౌరవంగా మాట్లాడాడు",        # Person with honorific
        "మేము విజయవాడ నగరం చూశాము",              # City sightseeing
    ]
    
    print(f"\n🧪 TESTING WITH {len(test_sentences)} SENTENCES")
    print("=" * 60)
    
    all_results = []
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Example {i}:")
        result = engine.analyze_sentence(sentence)
        all_results.append(result)
    
    # Show final statistics
    print(f"\n📊 FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    stats = engine.get_performance_stats()
    for key, value in stats.items():
        if key == 'module_usage':
            print(f"📈 Module Usage:")
            for module, count in value.items():
                print(f"   - {module}: {count} times")
        else:
            print(f"📈 {key.replace('_', ' ').title()}: {value}")
    
    # Save learned knowledge
    engine.save_knowledge()
    
    print(f"\n✅ ANALYSIS COMPLETED!")
    print(f"💡 The engine has processed {stats['total_predictions']} predictions")
    print(f"🎯 Average confidence: {stats['average_confidence']:.2f}")

if __name__ == "__main__":
    main()
