import re
from typing import Dict, List, Optional, Tuple, Any

# ==========================================
# ENHANCED ADVERB ENGINE (HOW & WHEN?)
# ==========================================
class EnhancedAdverbEngine:
    def __init__(self):
        # 1. MANNER ADVERBS (-ga suffix patterns) - Source 15, 16, 17
        self.manner_patterns = [
            (r'ంగా$', 'ం', 0.95),    # Veganga -> Vegam (Speedily)
            (r'గా$', '', 0.90),      # Mellaga -> Mella (Slowly)
            (r'చే$', '', 0.85),      # Swantah-che (Voluntarily)
            (r'తో$', '', 0.80),      # Premato (Lovingly)
        ]
        
        # 2. TIME ADVERBS - Comprehensive list
        self.time_adverbs = {
            'ఇప్పుడు': ('now', 'present_time'),
            'అప్పుడు': ('then', 'past_time'), 
            'రేపు': ('tomorrow', 'future_time'),
            'నిన్న': ('yesterday', 'past_time'),
            'ఈరోజు': ('today', 'present_time'),
            'ఎల్లప్పుడూ': ('always', 'frequency'),
            'తరచుగా': ('often', 'frequency'),
            'కొన్నిసార్లు': ('sometimes', 'frequency'),
            'అరుదుగా': ('rarely', 'frequency'),
            'ఇంకా': ('still', 'continuation'),
            'ఇప్పటికీ': ('still', 'continuation'),
            'వెంటనే': ('immediately', 'immediacy'),
            'త్వరలో': ('soon', 'imminent_future'),
            'ముందే': ('beforehand', 'anteriority'),
            'తర్వాత': ('after', 'posteriority'),
        }
        
        # 3. PLACE ADVERBS
        self.place_adverbs = {
            'ఇక్కడ': ('here', 'proximal'),
            'అక్కడ': ('there', 'distal'), 
            'ఎక్కడ': ('where', 'interrogative'),
            'ఎక్కడైనా': ('anywhere', 'indefinite'),
            'అన్నిచోట్ల': ('everywhere', 'universal'),
            'దగ్గర': ('near', 'proximity'),
            'దూరం': ('far', 'distance'),
            'పైన': ('above', 'vertical'),
            'కింద': ('below', 'vertical'),
            'లోపల': ('inside', 'containment'),
            'బయట': ('outside', 'exterior'),
            'ముందు': ('front', 'anterior'),
            'వెనుక': ('back', 'posterior'),
        }
        
        # 4. QUANTITY/DEGREE ADVERBS
        self.quantity_adverbs = {
            'చాలా': ('very', 'high_degree'),
            'కొంచెం': ('little', 'low_degree'),
            'మరింత': ('more', 'comparative'),
            'తక్కువ': ('less', 'comparative'),
            'పూర్తిగా': ('completely', 'totality'),
            'అంతా': ('entirely', 'totality'),
            'సగం': ('half', 'partial'),
            'కేవలం': ('only', 'exclusivity'),
            'మాత్రమే': ('only', 'exclusivity'),
        }
        
        # 5. FREQUENCY ADVERBS
        self.frequency_adverbs = {
            'ఎప్పుడూ': 'always',
            'తరచుగా': 'often', 
            'సాధారణంగా': 'usually',
            'కొన్నిసార్లు': 'sometimes',
            'అప్పుడప్పుడు': 'occasionally',
            'అరుదుగా': 'rarely',
            'ఎప్పుడూ కాదు': 'never'
        }

    def analyze(self, word: str) -> Optional[Dict[str, Any]]:
        word = word.strip()
        
        # Check Time Adverbs
        if word in self.time_adverbs:
            eng, subtype = self.time_adverbs[word]
            return {
                'type': 'ADVERB_TIME',
                'root': word,
                'subtype': subtype,
                'english': eng,
                'description': f'Time (When) - {eng}',
                'confidence': 0.95
            }

        # Check Place Adverbs
        if word in self.place_adverbs:
            eng, subtype = self.place_adverbs[word]
            return {
                'type': 'ADVERB_PLACE', 
                'root': word,
                'subtype': subtype,
                'english': eng,
                'description': f'Place (Where) - {eng}',
                'confidence': 0.95
            }

        # Check Quantity Adverbs
        if word in self.quantity_adverbs:
            eng, subtype = self.quantity_adverbs[word]
            return {
                'type': 'ADVERB_QUANTITY',
                'root': word, 
                'subtype': subtype,
                'english': eng,
                'description': f'Quantity/Degree (How much) - {eng}',
                'confidence': 0.95
            }

        # Check Frequency Adverbs
        if word in self.frequency_adverbs:
            eng = self.frequency_adverbs[word]
            return {
                'type': 'ADVERB_FREQUENCY',
                'root': word,
                'subtype': 'frequency',
                'english': eng,
                'description': f'Frequency (How often) - {eng}',
                'confidence': 0.95
            }

        # Check Manner Adverbs (-ga suffix)
        for pattern, replacement, confidence in self.manner_patterns:
            if re.search(pattern, word):
                root = re.sub(pattern, replacement, word)
                return {
                    'type': 'ADVERB_MANNER',
                    'root': root,
                    'subtype': 'derived_manner',
                    'english': f'in {root} manner',
                    'description': f'Manner (How) - Derived from "{root}"',
                    'confidence': confidence
                }

        return None

# ==========================================
# ENHANCED ADJECTIVE ENGINE (WHAT KIND?)
# ==========================================
class EnhancedAdjectiveEngine:
    def __init__(self):
        # 1. PURE ADJECTIVES (No derivation needed)
        self.pure_adjectives = {
            'మంచి': ('good', 'quality'),
            'చెడు': ('bad', 'quality'),
            'పెద్ద': ('big', 'size'),
            'చిన్న': ('small', 'size'),
            'కొత్త': ('new', 'age'),
            'పాత': ('old', 'age'),
            'ఎర్ర': ('red', 'color'),
            'నీలం': ('blue', 'color'),
            'పచ్చ': ('green', 'color'),
            'నల్ల': ('black', 'color'),
            'తెల్ల': ('white', 'color'),
            'బంగారు': ('golden', 'color'),
            'వేడి': ('hot', 'temperature'),
            'చలి': ('cold', 'temperature'),
            'తీవ్రమైన': ('intense', 'intensity'),
            'సాధారణ': ('ordinary', 'quality'),
        }
        
        # 2. DERIVED ADJECTIVE PATTERNS
        self.derived_patterns = [
            # -aina suffix (Source 14): Noun + aina = Adjective
            (r'ైన$', 'aina', 0.90, 'quality'),  # Telivaina (Intelligent)
            
            # -gala suffix: Possessive quality
            (r'గల$', 'gala', 0.85, 'possessive'),  # Dayagala (Kind)
            
            # -ni suffix: Quality descriptor  
            (r'ని$', 'ni', 0.80, 'quality'),  # Tiyani (Sweet)
            
            # -maya suffix: Made of/composed of
            (r'మయ$', 'maya', 0.75, 'composition'),  # Suvarnamaya (Golden)
            
            # -rukula suffix: Full of
            (r'రుకుల$', 'rukula', 0.70, 'abundance'),  # Puspitakula (Flowered)
        ]
        
        # 3. COMPARATIVE/SUPERLATIVE PATTERNS
        self.comparison_patterns = [
            (r'ఇంచుమించు$', 'approximate', 0.85),  # Almost/nearly
            (r'కంటె$', 'comparative', 0.80),       # More than
            (r'అత్యంత$', 'superlative', 0.90),     # Most
        ]

    def analyze(self, word: str) -> Optional[Dict[str, Any]]:
        word = word.strip()

        # Check Pure Adjectives
        if word in self.pure_adjectives:
            eng, subtype = self.pure_adjectives[word]
            return {
                'type': 'ADJECTIVE_PURE',
                'root': word,
                'subtype': subtype,
                'english': eng,
                'description': f'Quality/Attribute - {eng}',
                'confidence': 0.95
            }

        # Check Derived Adjectives
        for pattern, suffix, confidence, subtype in self.derived_patterns:
            if re.search(pattern, word):
                root = re.sub(pattern, '', word)
                suffix_desc = {
                    'aina': 'quality descriptor',
                    'gala': 'possessing quality', 
                    'ni': 'quality attribute',
                    'maya': 'composed of',
                    'rukula': 'full of'
                }
                
                return {
                    'type': 'ADJECTIVE_DERIVED',
                    'root': root,
                    'suffix': suffix,
                    'subtype': subtype,
                    'english': f'derived from {root}',
                    'description': f'Derived adjective ({suffix_desc.get(suffix, "unknown")})',
                    'confidence': confidence
                }

        # Check Comparative/Superlative
        for pattern, comp_type, confidence in self.comparison_patterns:
            if re.search(pattern, word):
                return {
                    'type': 'ADJECTIVE_COMPARISON',
                    'root': word,
                    'subtype': comp_type,
                    'english': f'{comp_type} form',
                    'description': f'Comparison ({comp_type})',
                    'confidence': confidence
                }

        return None

# ==========================================
# ENHANCED CONNECTOR ENGINE (COMPLEX SENTENCES)
# ==========================================
class EnhancedConnectorEngine:
    def __init__(self):
        # 1. COORDINATING CONJUNCTIONS
        self.coordinating = {
            'మరియు': ('and', 'addition'),
            'కానీ': ('but', 'contrast'),
            'లేదా': ('or', 'alternative'),
            'కాబట్టి': ('so', 'result'),
            'అందువల్ల': ('therefore', 'consequence'),
            'అయినా': ('yet', 'concession'),
        }
        
        # 2. SUBORDINATING CONJUNCTIONS
        self.subordinating = {
            'ఎందుకంటే': ('because', 'reason'),
            'వల్ల': ('due to', 'cause'),
            'అయితే': ('if', 'condition'),
            'చేత': ('by', 'instrument'),
            'గాని': ('although', 'concession'),
            'తర్వాత': ('after', 'time'),
            'ముందు': ('before', 'time'),
        }
        
        # 3. CONDITIONAL VERB SUFFIXES
        self.conditional_patterns = [
            (r'తే$', 'present_conditional', 0.90),  # Vastē (if comes)
            (r'టే$', 'present_conditional', 0.90),  # Tintē (if eats)
            (r'నచో$', 'conditional', 0.85),        # Vastēnē (if comes)
            (r'గా$', 'conditional', 0.80),         # Vastēgā (if comes)
        ]

    def analyze(self, word: str) -> Optional[Dict[str, Any]]:
        word = word.strip()

        # Check Coordinating Conjunctions
        if word in self.coordinating:
            eng, subtype = self.coordinating[word]
            return {
                'type': 'CONNECTOR_COORDINATING',
                'root': word,
                'subtype': subtype,
                'english': eng,
                'description': f'Coordinating conjunction - {eng}',
                'confidence': 0.95
            }

        # Check Subordinating Conjunctions
        if word in self.subordinating:
            eng, subtype = self.subordinating[word]
            return {
                'type': 'CONNECTOR_SUBORDINATING', 
                'root': word,
                'subtype': subtype,
                'english': eng,
                'description': f'Subordinating conjunction - {eng}',
                'confidence': 0.95
            }

        # Check Conditional Verb Forms
        for pattern, cond_type, confidence in self.conditional_patterns:
            if re.search(pattern, word):
                root = re.sub(pattern, '', word)
                return {
                    'type': 'CONNECTOR_CONDITIONAL',
                    'root': root,
                    'subtype': cond_type,
                    'english': f'if {root}',
                    'description': f'Conditional form - {cond_type}',
                    'confidence': confidence
                }

        return None

# ==========================================
# ENHANCED POSTPOSITION ENGINE
# ==========================================
class PostpositionEngine:
    def __init__(self):
        self.postpositions = {
            'లో': ('in', 'location'),
            'కు': ('to', 'dative'),
            'నుండి': ('from', 'ablative'),
            'తో': ('with', 'instrumental'),
            'కోసం': ('for', 'benefactive'),
            'గురించి': ('about', 'topic'),
            'వరకు': ('until', 'limit'),
            'ద్వారా': ('through', 'medium'),
            'పై': ('on', 'surface'),
            'కింద': ('under', 'position'),
            'ముందు': ('before', 'position'),
            'వెనుక': ('behind', 'position'),
            'సమీపంలో': ('near', 'proximity'),
        }

    def analyze(self, word: str) -> Optional[Dict[str, Any]]:
        if word in self.postpositions:
            eng, subtype = self.postpositions[word]
            return {
                'type': 'POSTPOSITION',
                'root': word,
                'subtype': subtype,
                'english': eng,
                'description': f'Postposition - {eng}',
                'confidence': 0.95
            }
        return None

# ==========================================
# MASTER MODIFIER ENGINE WITH ENSEMBLE VOTING
# ==========================================
class PowerModifierEngine:
    def __init__(self):
        self.adverb = EnhancedAdverbEngine()
        self.adjective = EnhancedAdjectiveEngine()
        self.connector = EnhancedConnectorEngine()
        self.postposition = PostpositionEngine()
        
        # Priority order for analysis
        self.analyzers = [
            ('postposition', self.postposition.analyze),
            ('connector', self.connector.analyze),
            ('adverb', self.adverb.analyze),
            ('adjective', self.adjective.analyze),
        ]

    def process(self, word: str) -> Dict[str, Any]:
        """Process word with ensemble approach"""
        candidates = []
        
        # Get analyses from all engines
        for analyzer_name, analyzer_func in self.analyzers:
            result = analyzer_func(word)
            if result:
                result['analyzer'] = analyzer_name
                candidates.append(result)
        
        # Return best candidate (highest confidence)
        if candidates:
            best_candidate = max(candidates, key=lambda x: x.get('confidence', 0))
            best_candidate['all_candidates'] = candidates
            return best_candidate
        
        # Unknown word
        return {
            'type': 'UNKNOWN',
            'root': word,
            'description': 'Unknown modifier type',
            'confidence': 0.0,
            'analyzer': 'none'
        }

    def batch_process(self, words: List[str]) -> List[Dict[str, Any]]:
        """Process multiple words efficiently"""
        return [self.process(word) for word in words]

    def analyze_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Analyze all modifiers in a sentence"""
        # Simple tokenization for Telugu
        words = re.findall(r'[\u0C00-\u0C7F]+', sentence)
        return self.batch_process(words)

# ==========================================
# COMPREHENSIVE TESTING
# ==========================================
def main():
    engine = PowerModifierEngine()
    
    # Comprehensive test cases
    test_words = [
        # Adverbs
        "వేగంగా", "మెల్లగా", "ఇప్పుడు", "ఇక్కడ", "చాలా", "తరచుగా",
        
        # Adjectives  
        "మంచి", "తెలివైన", "తీయని", "సువర్ణమయ", "అత్యంత",
        
        # Connectors
        "మరియు", "కానీ", "ఎందుకంటే", "వస్తే", "అయితే",
        
        # Postpositions
        "లో", "కు", "తో", "కోసం",
        
        # Edge cases
        "ప్రేమతో", "సంతోషంగా", "విజయవంతంగా"
    ]
    
    print("🧪 COMPREHENSIVE TELUGU MODIFIER ANALYSIS")
    print("=" * 90)
    print(f"{'WORD':<15} | {'TYPE':<25} | {'SUBTYPE':<15} | {'ROOT':<12} | {'CONF':<5} | {'DESCRIPTION'}")
    print("-" * 90)
    
    results = engine.batch_process(test_words)
    
    for result in results:
        word = result.get('root', '') if result['type'] == 'UNKNOWN' else test_words[results.index(result)]
        print(f"{word:<15} | {result['type']:<25} | {result.get('subtype', ''):<15} | {result.get('root', ''):<12} | {result.get('confidence', 0):<5.2f} | {result.get('description', '')}")
    
    # Sentence analysis demo
    print(f"\n📝 SENTENCE ANALYSIS DEMO:")
    print("=" * 90)
    test_sentence = "రాము వేగంగా మరియు చురుకుగా పని చేస్తాడు కానీ అతను మెల్లగా మాట్లాడతాడు"
    sentence_results = engine.analyze_sentence(test_sentence)
    
    print(f"Sentence: {test_sentence}")
    print("-" * 90)
    for result in sentence_results:
        if result['type'] != 'UNKNOWN':
            print(f"  {result['root']:<12} -> {result['type']:<20} ({result.get('subtype', '')})")

if __name__ == "__main__":
    main()
