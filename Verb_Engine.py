import re
import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

# ==========================================
# MODULE A: ADVANCED VERB MORPHOLOGY ANALYZER
# ==========================================
class AdvancedVerbMorphologyAnalyzer:
    def __init__(self):
        # Enhanced patterns with better coverage and confidence scoring
        self.person_patterns = [
            # 1st Person
            (r'న్ను$', '1st_Singular', 0.95),    # -nnu (variant)
            (r'ను$', '1st_Singular', 0.90),      # -nu (Nenu vellanu)
            (r'మ్ము$', '1st_Plural', 0.85),      # -mmu (Memu chestammu)
            (r'ము$', '1st_Plural', 0.90),       # -mu (Memu vellamu)
            
            # 2nd Person  
            (r'వ్వు$', '2nd_Singular', 0.85),    # -vvu (Nuvvu chestavvu)
            (r'వు$', '2nd_Singular', 0.90),      # -vu (Nuvvu vellavu)
            (r'ర్రు$', '2nd_Plural', 0.80),      # -rru (Meeru chestarru)
            (r'రు$', '2nd_Plural', 0.90),        # -ru (Meeru vellaru)
            
            # 3rd Person
            (r'ండు$', '3rd_Male_Singular', 0.85), # -ndu (Atanu chestaadu)
            (r'డు$', '3rd_Male_Singular', 0.90),  # -du (Atanu velladu)
            (r'ద్దు$', '3rd_Fem_Singular', 0.80), # -ddu (Ame chestaddu)
            (r'ది$', '3rd_Fem_Neut', 0.90),       # -di (Ame/Adi vellindi)
            (r'యి$', '3rd_Neut_Plural', 0.85),   # -yi (Avi chestayi)
            (r'వి$', '3rd_Neut_Plural', 0.90),   # -vi (Avi vellayi)
        ]

        # Enhanced tense/mode patterns with better ordering
        self.tense_patterns = [
            # Complex forms first
            (r'తున్నారు$', 'Present_Continuous_Plural', 0.95),
            (r'టున్నారు$', 'Present_Continuous_Plural', 0.95),
            (r'తున్నాడు$', 'Present_Continuous_Male', 0.95),
            (r'టున్నాడు$', 'Present_Continuous_Male', 0.95),
            (r'తున్నాను$', 'Present_Continuous_Singular', 0.95),
            (r'టున్నాను$', 'Present_Continuous_Singular', 0.95),
            (r'తున్నా$', 'Present_Continuous', 0.90),
            (r'టున్నా$', 'Present_Continuous', 0.90),
            
            # Future tense
            (r'తారు$', 'Future_Plural', 0.90),
            (r'టారు$', 'Future_Plural', 0.90),
            (r'తాడు$', 'Future_Male', 0.90),
            (r'టాడు$', 'Future_Male', 0.90),
            (r'తాను$', 'Future_Singular', 0.90),
            (r'టాను$', 'Future_Singular', 0.90),
            (r'తా$', 'Future', 0.85),
            (r'టా$', 'Future', 0.85),
            
            # Past tense variants
            (r'శారు$', 'Past_Plural', 0.90),
            (r'సారు$', 'Past_Plural', 0.90),
            (r'శాడు$', 'Past_Male', 0.90),
            (r'సాడు$', 'Past_Male', 0.90),
            (r'శాను$', 'Past_Singular', 0.90),
            (r'సాను$', 'Past_Singular', 0.90),
            (r'శా$', 'Past', 0.85),
            (r'సా$', 'Past', 0.85),
            (r'ఇన్నారు$', 'Past_Plural', 0.85),
            (r'ఇన్నాడు$', 'Past_Male', 0.85),
            (r'ఇన్నాను$', 'Past_Singular', 0.85),
            (r'ఇన్నా$', 'Past', 0.80),
            
            # Negation patterns
            (r'లేదు$', 'Negative_Past', 0.95),
            (r'లేము$', 'Negative_Past_Plural', 0.95),
            (r'లేను$', 'Negative_Past_Singular', 0.95),
            (r'కూడదు$', 'Negative_Permissive', 0.90),
            (r'వద్దు$', 'Negative_Imperative', 0.95),
            (r'కు$', 'Negative_Imperative', 0.85),
            
            # Modality markers
            (r'వచ్చు$', 'Potential', 0.90),
            (r'లేకపోయింది$', 'Failed_Action', 0.95),
            (r'అలి$', 'Obligatory', 0.90),
            (r'యాలి$', 'Obligatory', 0.85),
            (r'యాల్సిందే$', 'Compulsive', 0.95),
            (r'చ్చు$', 'Permissive', 0.85),
            (r'ండి$', 'Honorific_Imperative', 0.95),
        ]

        # Causative and aspect markers
        self.aspect_patterns = [
            (r'ఇంచి$', 'Causative_Perfective', 0.90),
            (r'ఇంచ$', 'Causative', 0.85),
            (r'ఇప్పి$', 'Causative_Perfective', 0.90),
            (r'ఇప్ప$', 'Causative', 0.85),
            (r'కొని$', 'Completive_Aspect', 0.80),
            (r'పెట్టి$', 'Intensive_Aspect', 0.80),
        ]

    def analyze(self, verb: str) -> Tuple[str, Dict[str, Any]]:
        """Advanced verb analysis with confidence scoring"""
        original_verb = verb.strip()
        current_form = original_verb
        
        analysis = {
            "person": "Unknown",
            "tense": "Root/Imperative/Infinitive",
            "aspect": "Simple",
            "polarity": "Positive",
            "mood": "Indicative",
            "is_causative": False,
            "markers_found": [],
            "confidence": 1.0,
            "analysis_steps": []
        }

        # Track transformations for debugging
        steps = []

        # Phase 1: Aspect and Voice Markers (innermost)
        for pattern, aspect, confidence in self.aspect_patterns:
            if re.search(pattern, current_form):
                match = re.search(pattern, current_form)
                steps.append({
                    'phase': 'aspect',
                    'from': current_form,
                    'pattern': pattern,
                    'feature': aspect,
                    'confidence': confidence
                })
                
                analysis["aspect"] = aspect
                if "Causative" in aspect:
                    analysis["is_causative"] = True
                
                current_form = re.sub(pattern, '', current_form)
                analysis["confidence"] *= confidence
                analysis["markers_found"].append(aspect)
                break

        # Phase 2: Tense/Mode Markers
        for pattern, tense, confidence in self.tense_patterns:
            if re.search(pattern, current_form):
                steps.append({
                    'phase': 'tense',
                    'from': current_form,
                    'pattern': pattern,
                    'feature': tense,
                    'confidence': confidence
                })
                
                analysis["tense"] = tense
                
                # Set polarity and mood
                if "Negative" in tense:
                    analysis["polarity"] = "Negative"
                if "Imperative" in tense:
                    analysis["mood"] = "Imperative"
                if "Potential" in tense:
                    analysis["mood"] = "Potential"
                    
                current_form = re.sub(pattern, '', current_form)
                analysis["confidence"] *= confidence
                analysis["markers_found"].append(tense)
                break

        # Phase 3: Person Markers (outermost)
        for pattern, person, confidence in self.person_patterns:
            if re.search(pattern, current_form):
                steps.append({
                    'phase': 'person',
                    'from': current_form,
                    'pattern': pattern,
                    'feature': person,
                    'confidence': confidence
                })
                
                analysis["person"] = person
                current_form = re.sub(pattern, '', current_form)
                analysis["confidence"] *= confidence
                analysis["markers_found"].append(person)
                break

        analysis["analysis_steps"] = steps
        analysis["confidence"] = round(analysis["confidence"], 3)
        
        return current_form, analysis

# ==========================================
# MODULE B: INTELLIGENT ROOT NORMALIZER
# ==========================================
class IntelligentVerbRootDictionary:
    def __init__(self, dictionary_path: str = "data/verb_dictionary.json"):
        self.dictionary_path = dictionary_path
        self.root_map = {}
        self.verb_classes = {}
        self.conjugation_patterns = {}
        
        self.load_base_dictionary()
        self.load_external_dictionary()
        self.initialize_verb_classes()

    def load_base_dictionary(self):
        """Load comprehensive base dictionary"""
        self.root_map = {
            # Class 1: -cu verbs
            'వచ్చ': 'వచ్చు', 'వచ్': 'వచ్చు', 'రా': 'రా',
            
            # Class 2: -yu verbs  
            'చేస': 'చేయు', 'చేశ': 'చేయు', 'చేయ': 'చేయు',
            'చేస్త': 'చేయు', 'ఆడ': 'ఆడు', 'ఆట': 'ఆడు',
            
            # Class 3: -nu verbs
            'తిన్న': 'తిను', 'తిం': 'తిను', 'తిన': 'తిను',
            'కొన్న': 'కొను', 'కొం': 'కొను', 'కొన': 'కొను',
            'విన్న': 'విను', 'విం': 'విను', 'విన': 'విను',
            
            # Class 4: -du verbs
            'చదివ': 'చదువు', 'చది': 'చదువు',
            'నడిచ': 'నడుచు', 'నడి': 'నడుచు',
            
            # Class 5: -tu verbs
            'ఎక్క': 'ఎక్కు', 'పడ': 'పడు', 'కూర్చొ': 'కూర్చొను',
            
            # Class 6: Irregular
            'ఉన్న': 'ఉండు', 'ఉం': 'ఉండు', 'ఉన్': 'ఉండు',
            'వెళ్ళ': 'వెళ్ళు', 'వెళ్': 'వెళ్ళు', 'వెళ్ల': 'వెళ్ళు',
            'పోయ': 'పోవు', 'పో': 'పోవు',
            'ఇచ్చ': 'ఇచ్చు', 'ఇస': 'ఇచ్చు',
        }

    def initialize_verb_classes(self):
        """Initialize verb conjugation classes"""
        self.verb_classes = {
            'వచ్చు': {'class': 'irregular', 'pattern': 'cu'},
            'చేయు': {'class': 'yu_verb', 'pattern': 'yu'},
            'తిను': {'class': 'nu_verb', 'pattern': 'nu'},
            'చదువు': {'class': 'du_verb', 'pattern': 'du'},
            'ఉండు': {'class': 'irregular', 'pattern': 'du'},
            'వెళ్ళు': {'class': 'irregular', 'pattern': 'lu'},
        }

    def load_external_dictionary(self):
        """Load external dictionary if available"""
        try:
            if os.path.exists(self.dictionary_path):
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    external_data = json.load(f)
                    self.root_map.update(external_data.get('root_map', {}))
        except Exception as e:
            print(f"Warning: Could not load external dictionary: {e}")

    def normalize(self, raw_stem: str, analysis: Dict = None) -> Dict[str, Any]:
        """Intelligent root normalization with pattern matching"""
        stem = raw_stem.strip()
        
        result = {
            'raw_stem': stem,
            'normalized_root': stem,
            'verb_class': 'unknown',
            'confidence': 0.7,
            'normalization_method': 'direct'
        }

        # 1. Direct dictionary lookup (highest confidence)
        if stem in self.root_map:
            result['normalized_root'] = self.root_map[stem]
            result['confidence'] = 0.95
            result['normalization_method'] = 'dictionary'
        
        # 2. Pattern-based normalization
        else:
            # Common stem transformations
            transformations = [
                # Remove common imperfective markers
                (r'త్త$', '', 0.8),  # Chesta -> Ches
                (r'ట్ట$', '', 0.8),  # Tinta -> Tin
                (r'ద్ద$', '', 0.8),  # Addama -> Ada
                
                # Handle gemination
                (r'([క-హ])\1$', r'\1', 0.7),  # Double consonants
                
                # Add common verb endings
                (r'([క-హ])$', r'\1ు', 0.6),   # Add -u ending
            ]
            
            for pattern, replacement, conf in transformations:
                if re.search(pattern, stem):
                    candidate = re.sub(pattern, replacement, stem)
                    if len(candidate) >= 2:  # Reasonable root length
                        result['normalized_root'] = candidate
                        result['confidence'] *= conf
                        result['normalization_method'] = 'pattern_based'
                        break

        # 3. Determine verb class
        root = result['normalized_root']
        if root in self.verb_classes:
            result['verb_class'] = self.verb_classes[root]['class']
        else:
            # Infer class from ending
            if root.endswith('యు'):
                result['verb_class'] = 'yu_verb'
            elif root.endswith('ను'):
                result['verb_class'] = 'nu_verb' 
            elif root.endswith('చ్చు'):
                result['verb_class'] = 'cu_verb'
            elif root.endswith('డు'):
                result['verb_class'] = 'du_verb'
            else:
                result['verb_class'] = 'regular'

        result['confidence'] = round(result['confidence'], 3)
        return result

# ==========================================
# MODULE C: CONTEXT-AWARE VERB ENGINE
# ==========================================
class ContextAwareVerbEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.analyzer = AdvancedVerbMorphologyAnalyzer()
        self.dictionary = IntelligentVerbRootDictionary()
        
        # Statistics and learning
        self.usage_stats = defaultdict(int)
        self.learned_verbs = {}
        
        # Load learned data
        self.load_learned_data()

    def load_learned_data(self):
        """Load previously learned verb patterns"""
        try:
            learned_file = os.path.join(self.data_dir, 'learned_verbs.json')
            if os.path.exists(learned_file):
                with open(learned_file, 'r', encoding='utf-8') as f:
                    self.learned_verbs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load learned data: {e}")

    def save_learned_data(self):
        """Save learned verb patterns"""
        try:
            learned_file = os.path.join(self.data_dir, 'learned_verbs.json')
            with open(learned_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_verbs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save learned data: {e}")

    def process_verb(self, raw_verb: str, context: List[str] = None) -> Dict[str, Any]:
        """Process verb with contextual analysis"""
        
        # Update usage statistics
        self.usage_stats[raw_verb] += 1
        
        # 1. Morphological analysis
        stem, analysis = self.analyzer.analyze(raw_verb)
        
        # 2. Root normalization
        root_info = self.dictionary.normalize(stem, analysis)
        
        # 3. Generate comprehensive description
        description = self._generate_detailed_description(root_info, analysis)
        
        # 4. Contextual refinement
        if context:
            analysis['context'] = self._analyze_context(raw_verb, context)
        
        # 5. Learn from this analysis
        self._learn_verb_pattern(raw_verb, root_info, analysis)
        
        return {
            "original": raw_verb,
            "stem": stem,
            "root": root_info['normalized_root'],
            "root_info": root_info,
            "morphology": analysis,
            "description": description,
            "overall_confidence": round(analysis['confidence'] * root_info['confidence'], 3)
        }

    def _generate_detailed_description(self, root_info: Dict, analysis: Dict) -> str:
        """Generate human-readable description"""
        parts = []
        
        # Root and action
        parts.append(f"Action: {root_info['normalized_root']}")
        
        # Tense description
        tense_desc = {
            'Present_Continuous': "happening now",
            'Past': "completed", 
            'Future': "will happen",
            'Negative_Past': "did not happen",
            'Negative_Imperative': "don't do!",
            'Obligatory': "must do",
            'Compulsive': "should do",
            'Honorific_Imperative': "please do",
            'Potential': "can do"
        }
        
        tense = analysis['tense']
        if tense in tense_desc:
            parts.append(tense_desc[tense])
        
        # Person information
        if analysis['person'] != 'Unknown':
            parts.append(f"by {analysis['person'].replace('_', ' ').lower()}")
        
        # Aspect information
        if analysis['is_causative']:
            parts.append("(causative - made someone do)")
        if analysis['aspect'] != 'Simple':
            parts.append(f"({analysis['aspect'].replace('_', ' ').lower()})")
        
        # Add verb class info
        parts.append(f"[{root_info['verb_class']} class]")
        
        return " | ".join(parts)

    def _analyze_context(self, verb: str, context: List[str]) -> Dict[str, Any]:
        """Analyze verb in sentence context"""
        context_analysis = {
            'position_in_sentence': None,
            'likely_subject': None,
            'likely_object': None,
            'sentence_type': 'declarative'
        }
        
        # Simple context analysis
        words = [w for w in context if w.strip()]
        verb_index = words.index(verb) if verb in words else -1
        
        if verb_index >= 0:
            context_analysis['position_in_sentence'] = verb_index
            
            # Look for subject before verb
            if verb_index > 0:
                context_analysis['likely_subject'] = words[verb_index - 1]
            
            # Look for object after verb  
            if verb_index < len(words) - 1:
                context_analysis['likely_object'] = words[verb_index + 1]
                
            # Detect question
            if any(q in context for q in ['ఏమి', 'ఎందుకు', 'ఎప్పుడు', 'ఎవరు']):
                context_analysis['sentence_type'] = 'interrogative'
        
        return context_analysis

    def _learn_verb_pattern(self, original: str, root_info: Dict, analysis: Dict):
        """Learn from successful analyses"""
        key = f"{root_info['normalized_root']}_{analysis['tense']}_{analysis['person']}"
        
        if key not in self.learned_verbs:
            self.learned_verbs[key] = {
                'root': root_info['normalized_root'],
                'tense': analysis['tense'],
                'person': analysis['person'],
                'examples': [],
                'frequency': 0
            }
        
        self.learned_verbs[key]['examples'].append(original)
        self.learned_verbs[key]['frequency'] += 1
        
        # Keep only recent examples
        if len(self.learned_verbs[key]['examples']) > 5:
            self.learned_verbs[key]['examples'] = self.learned_verbs[key]['examples'][-5:]

    def batch_process(self, verbs: List[str]) -> List[Dict[str, Any]]:
        """Process multiple verbs efficiently"""
        return [self.process_verb(verb) for verb in verbs]

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_verbs_processed': sum(self.usage_stats.values()),
            'unique_verbs_processed': len(self.usage_stats),
            'learned_patterns': len(self.learned_verbs),
            'most_common_verbs': dict(sorted(self.usage_stats.items(), 
                                           key=lambda x: x[1], reverse=True)[:5])
        }

# ==========================================
# ENHANCED TESTING AND USAGE
# ==========================================
def main():
    """Comprehensive testing of the enhanced verb engine"""
    
    print("🚀 ENHANCED TELUGU VERB MORPHOLOGY ANALYZER")
    print("=" * 80)
    
    # Initialize the engine
    engine = ContextAwareVerbEngine()
    
    # Test cases covering various Telugu verb patterns
    test_verbs = [
        "తింటున్నాను",      # Present Continuous
        "తిన్నాను",         # Simple Past  
        "తినలేదు",          # Negative Past
        "వెళ్లకండి",         # Negative Imperative
        "తినిపించాను",      # Causative Past
        "రాశాను",           # Irregular Past (Write)
        "చేయాల్సిందే",      # Compulsive
        "కూర్చోండి",        # Honorific Imperative
        "వస్తాను",          # Future
        "చేయగలను",          # Potential
        "పోతున్నాను",        # Going (Aspect)
        "ఉండేవాడు",         # Habitual Past
    ]
    
    print(f"\n{'VERB':<15} | {'ROOT':<10} | {'TENSE':<20} | {'PERSON':<15} | {'CONF':<5} | {'DESCRIPTION'}")
    print("-" * 120)
    
    results = []
    for verb in test_verbs:
        result = engine.process_verb(verb)
        results.append(result)
        
        print(f"{result['original']:<15} | {result['root']:<10} | {result['morphology']['tense']:<20} | {result['morphology']['person']:<15} | {result['overall_confidence']:<5} | {result['description']}")
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"\n📊 ENGINE STATISTICS:")
    print(f"   Total verbs processed: {stats['total_verbs_processed']}")
    print(f"   Unique verbs: {stats['unique_verbs_processed']}")
    print(f"   Learned patterns: {stats['learned_patterns']}")
    
    # Save learned data
    engine.save_learned_data()
    
    print(f"\n✅ ANALYSIS COMPLETED!")

if __name__ == "__main__":
    main()
