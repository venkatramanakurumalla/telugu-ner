import re
from typing import Dict, List, Optional, Tuple, Any

class AdvancedPronounEngine:
    def __init__(self):
        # 1. COMPREHENSIVE BASE FORMS (Nominative Case)
        self.base_forms = {
            # 1st Person
            'నేను': {'person': '1st', 'number': 'Singular', 'gender': 'Any', 'eng': 'I', 'formality': 'informal'},
            'నా': {'person': '1st', 'number': 'Singular', 'gender': 'Any', 'eng': 'I', 'formality': 'oblique_base'},
            'మేము': {'person': '1st', 'number': 'Plural', 'gender': 'Any', 'eng': 'We', 'formality': 'formal'},
            'మనము': {'person': '1st', 'number': 'Plural', 'gender': 'Any', 'eng': 'We', 'formality': 'inclusive'},
            'మన': {'person': '1st', 'number': 'Plural', 'gender': 'Any', 'eng': 'We', 'formality': 'oblique_inclusive'},
            
            # 2nd Person
            'నీవు': {'person': '2nd', 'number': 'Singular', 'gender': 'Any', 'eng': 'You', 'formality': 'formal'},
            'నువ్వు': {'person': '2nd', 'number': 'Singular', 'gender': 'Any', 'eng': 'You', 'formality': 'informal'},
            'నీ': {'person': '2nd', 'number': 'Singular', 'gender': 'Any', 'eng': 'You', 'formality': 'oblique_informal'},
            'మీరు': {'person': '2nd', 'number': 'Plural', 'gender': 'Any', 'eng': 'You', 'formality': 'respectful'},
            'మీ': {'person': '2nd', 'number': 'Plural', 'gender': 'Any', 'eng': 'You', 'formality': 'oblique_respectful'},
            'తమరు': {'person': '2nd', 'number': 'Singular', 'gender': 'Any', 'eng': 'You', 'formality': 'high_respect'},
            'తమ': {'person': '2nd', 'number': 'Singular', 'gender': 'Any', 'eng': 'You', 'formality': 'oblique_high_respect'},
            
            # 3rd Person
            'అతను': {'person': '3rd', 'number': 'Singular', 'gender': 'Male', 'eng': 'He', 'formality': 'neutral'},
            'అతడు': {'person': '3rd', 'number': 'Singular', 'gender': 'Male', 'eng': 'He', 'formality': 'neutral'},
            'ఆమె': {'person': '3rd', 'number': 'Singular', 'gender': 'Female', 'eng': 'She', 'formality': 'neutral'},
            'అది': {'person': '3rd', 'number': 'Singular', 'gender': 'Neuter', 'eng': 'It/That', 'formality': 'neutral'},
            'ఇది': {'person': '3rd', 'number': 'Singular', 'gender': 'Neuter', 'eng': 'It/This', 'formality': 'neutral'},
            'వారు': {'person': '3rd', 'number': 'Plural', 'gender': 'Human', 'eng': 'They', 'formality': 'respectful'},
            'వాళ్ళు': {'person': '3rd', 'number': 'Plural', 'gender': 'Human', 'eng': 'They', 'formality': 'informal'},
            'అవి': {'person': '3rd', 'number': 'Plural', 'gender': 'NonHuman', 'eng': 'Those', 'formality': 'neutral'},
            'ఇవి': {'person': '3rd', 'number': 'Plural', 'gender': 'NonHuman', 'eng': 'These', 'formality': 'neutral'},
            
            # Reflexive
            'తాను': {'person': '3rd', 'number': 'Singular', 'gender': 'Any', 'eng': 'Self', 'formality': 'reflexive'},
            'తన': {'person': '3rd', 'number': 'Singular', 'gender': 'Any', 'eng': 'Self', 'formality': 'oblique_reflexive'},
            
            # Interrogative
            'ఎవరు': {'person': 'interrogative', 'number': 'Any', 'gender': 'Human', 'eng': 'Who', 'formality': 'neutral'},
            'ఏమి': {'person': 'interrogative', 'number': 'Any', 'gender': 'Neuter', 'eng': 'What', 'formality': 'neutral'},
            'ఎవడు': {'person': 'interrogative', 'number': 'Singular', 'gender': 'Male', 'eng': 'Who', 'formality': 'informal'},
            'ఎవతె': {'person': 'interrogative', 'number': 'Singular', 'gender': 'Female', 'eng': 'Who', 'formality': 'informal'},
            
            # Demonstrative
            'అదే': {'person': '3rd', 'number': 'Singular', 'gender': 'Neuter', 'eng': 'That itself', 'formality': 'emphatic'},
            'ఇదే': {'person': '3rd', 'number': 'Singular', 'gender': 'Neuter', 'eng': 'This itself', 'formality': 'emphatic'},
        }

        # 2. COMPREHENSIVE CASE MAPPINGS
        self.case_patterns = {
            # Accusative (Object) - ni/nu endings
            'Accusative': [
                (r'న్ను$', 'నేను'),  # Nannu (me)
                (r'మ్మల్ని$', 'మేము'), # Mammalni (us)
                (r'న్ను$', 'నువ్వు'), # Ninnu (you)
                (r'మ్మల్ని$', 'మీరు'), # Mimmalni (you plural)
                (r'న్ని$', 'అతను'),  # Atanni (him)
                (r'ను$', 'ఆమె'),    # Aamenu (her)
            ],
            
            # Dative (To/For) - ku/ki endings
            'Dative': [
                (r'కు$', 'నేను'),    # Naaku (to me)
                (r'కు$', 'మేము'),    # Maaku (to us)
                (r'కు$', 'నువ్వు'),  # Neeku (to you)
                (r'కు$', 'మీరు'),    # Meeku (to you plural)
                (r'కి$', 'అతను'),    # Ataniki (to him)
                (r'కి$', 'ఆమె'),     # Aameki (to her)
            ],
            
            # Genitive/Possessive (Of) - no specific ending, but oblique base
            'Genitive': [
                (r'^నా', 'నేను'),    # Naa (my)
                (r'^మా', 'మేము'),    # Maa (our)
                (r'^నీ', 'నువ్వు'),  # Nee (your)
                (r'^మీ', 'మీరు'),    # Mee (your plural)
                (r'^అతని', 'అతను'), # Atani (his)
                (r'^ఆమె', 'ఆమె'),    # Aame (her)
                (r'^తన', 'తాను'),    # Tana (his/her own)
            ],
            
            # Instrumental (With) - to/tho endings
            'Instrumental': [
                (r'తో$', 'నేను'),    # Naatho (with me)
                (r'తో$', 'మేము'),    # Maatho (with us)
                (r'తో$', 'నువ్వు'),  # Neetho (with you)
                (r'తో$', 'మీరు'),    # Meetho (with you plural)
            ],
            
            # Locative (In/At) - lo endings
            'Locative': [
                (r'లో$', 'నేను'),    # Naalo (in me)
                (r'లో$', 'మేము'),    # Maalo (in us)
                (r'లో$', 'నువ్వు'),  # Neelo (in you)
            ],
            
            # Ablative (From) - nundi endings
            'Ablative': [
                (r'నుండి$', 'నేను'), # Naanundi (from me)
                (r'నుండి$', 'మేము'), # Maanundi (from us)
                (r'నుండి$', 'నువ్వు'), # Neenundi (from you)
            ]
        }

        # 3. COMPOUND SUFFIXES (Postpositions that attach to oblique forms)
        self.compound_suffixes = {
            'కోసం': 'for',
            'వల్ల': 'because of', 
            'ద్వారా': 'through',
            'గురించి': 'about',
            'చెంది': 'regarding',
            'వరకు': 'until',
            'లాగా': 'like',
            'పొంది': 'having',
        }

    def detect_case(self, word: str) -> Tuple[Optional[str], Optional[str], float]:
        """Detect case and root with confidence scoring"""
        # Direct base form match (highest confidence)
        if word in self.base_forms:
            return 'Nominative', word, 0.98
        
        # Check each case pattern
        for case, patterns in self.case_patterns.items():
            for pattern, root_base in patterns:
                if case == 'Genitive':
                    # Genitive patterns are prefixes
                    if word.startswith(pattern.replace('^', '')):
                        remaining = word[len(pattern.replace('^', '')):]
                        # Check if remaining part is a compound suffix
                        if not remaining or remaining in self.compound_suffixes:
                            return case, root_base, 0.95
                else:
                    # Other cases are suffixes
                    if re.search(pattern, word):
                        root = re.sub(pattern, '', word)
                        # Verify this is a valid root
                        if root in [r.replace('^', '') for r in [p[1] for p in self.case_patterns[case]]]:
                            return case, root_base, 0.90
        
        # Check for compound forms (oblique + postposition)
        for oblique_root in ['నా', 'మా', 'నీ', 'మీ', 'అతని', 'ఆమె', 'తన']:
            if word.startswith(oblique_root) and len(word) > len(oblique_root):
                suffix = word[len(oblique_root):]
                if suffix in self.compound_suffixes:
                    # Map oblique root back to base
                    base_map = {
                        'నా': 'నేను', 'మా': 'మేము', 'నీ': 'నువ్వు', 
                        'మీ': 'మీరు', 'అతని': 'అతను', 'ఆమె': 'ఆమె', 'తన': 'తాను'
                    }
                    base = base_map.get(oblique_root, oblique_root)
                    return f"Genitive+{suffix}", base, 0.85
        
        return None, None, 0.0

    def analyze(self, word: str) -> Optional[Dict[str, Any]]:
        """Comprehensive pronoun analysis"""
        word = word.strip()
        
        # Detect case and root
        case, root, confidence = self.detect_case(word)
        
        if case and root and root in self.base_forms:
            base_info = self.base_forms[root].copy()
            
            # Enhanced description based on case
            case_descriptions = {
                'Nominative': 'Subject form',
                'Accusative': 'Object form (receives action)',
                'Dative': 'Indirect object (to/for)',
                'Genitive': 'Possessive form (of/belonging to)',
                'Instrumental': 'Instrumental (with/using)',
                'Locative': 'Locative (in/at/on)',
                'Ablative': 'Ablative (from)',
            }
            
            description = case_descriptions.get(case, f'{case} form')
            
            # Handle compound forms specially
            if '+' in case:
                base_case, suffix = case.split('+')
                description = f"{case_descriptions.get(base_case, base_case)} + '{suffix}' ({self.compound_suffixes.get(suffix, 'unknown')})"
                case = base_case
            
            return {
                'root': root,
                'type': 'PRONOUN',
                'case': case,
                'description': description,
                'details': base_info,
                'confidence': confidence,
                'english_equivalent': self._get_english_equivalent(root, case, base_info)
            }
        
        return None

    def _get_english_equivalent(self, root: str, case: str, base_info: Dict) -> str:
        """Generate English equivalent based on case"""
        base_eng = base_info['eng']
        
        case_equivalents = {
            'Nominative': base_eng,
            'Accusative': f'me/us/you' if base_eng in ['I', 'We', 'You'] else f'him/her/it',
            'Dative': f'to {base_eng.lower()}',
            'Genitive': f'my/our/your' if base_eng in ['I', 'We', 'You'] else f'his/her/its',
            'Instrumental': f'with {base_eng.lower()}',
            'Locative': f'in {base_eng.lower()}',
            'Ablative': f'from {base_eng.lower()}',
        }
        
        return case_equivalents.get(case, base_eng)

    def analyze_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Find and analyze all pronouns in a sentence"""
        # Simple Telugu tokenization
        words = re.findall(r'[\u0C00-\u0C7F]+', sentence)
        pronouns = []
        
        for word in words:
            analysis = self.analyze(word)
            if analysis:
                analysis['original'] = word
                pronouns.append(analysis)
        
        return pronouns

    def get_pronoun_paradigm(self, base_pronoun: str) -> Dict[str, str]:
        """Generate all case forms for a given base pronoun"""
        if base_pronoun not in self.base_forms:
            return {}
        
        paradigm = {'Nominative': base_pronoun}
        
        # Generate common case forms (simplified)
        base_map = {
            'నేను': 'నా', 'మేము': 'మా', 'నువ్వు': 'నీ', 
            'మీరు': 'మీ', 'అతను': 'అతని', 'ఆమె': 'ఆమె', 'తాను': 'తన'
        }
        
        oblique_base = base_map.get(base_pronoun, '')
        if oblique_base:
            paradigm.update({
                'Accusative': f'{oblique_base}న్ను' if base_pronoun in ['నేను', 'నువ్వు'] else f'{oblique_base}ను',
                'Dative': f'{oblique_base}కు',
                'Genitive': oblique_base,
                'Instrumental': f'{oblique_base}తో',
                'Locative': f'{oblique_base}లో',
            })
        
        return paradigm


# ==========================================
# COMPREHENSIVE TESTING
# ==========================================
def main():
    engine = AdvancedPronounEngine()
    
    # Comprehensive test cases
    test_words = [
        # Base forms
        "నేను", "మేము", "నువ్వు", "మీరు", "అతను", "ఆమె", "అది", "వారు",
        
        # Case forms
        "నాకు", "మాకు", "నీకు", "మీకు", "అతనికి", 
        "నన్ను", "మమ్మల్ని", "నిన్ను", "మిమ్మల్ని", "అతన్ని",
        "నా", "మా", "నీ", "మీ", "అతని", "తన",
        "నాతో", "మాతో", "నీతో", "మీతో",
        "నాలో", "మాలో", "నీలో",
        
        # Compound forms
        "నాకోసం", "మావల్ల", "అతనిద్వారా", "నాగురించి",
        
        # Reflexive and interrogative
        "తాను", "తన", "ఎవరు", "ఏమి", "ఎవడు",
        
        # Emphatic forms
        "నేనే", "అదే", "ఇదే"
    ]
    
    print("🧪 ADVANCED TELUGU PRONOUN ANALYZER")
    print("=" * 100)
    print(f"{'PRONOUN':<12} | {'ROOT':<10} | {'CASE':<20} | {'PERSON/NUMBER':<15} | {'ENGLISH':<20} | {'CONF'}")
    print("-" * 100)
    
    results = []
    for word in test_words:
        analysis = engine.analyze(word)
        if analysis:
            details = analysis['details']
            person_num = f"{details['person']}/{details['number']}"
            print(f"{word:<12} | {analysis['root']:<10} | {analysis['case']:<20} | {person_num:<15} | {analysis['english_equivalent']:<20} | {analysis['confidence']:.2f}")
            results.append(analysis)
    
    # Sentence analysis demo
    print(f"\n📝 SENTENCE ANALYSIS DEMO:")
    print("=" * 100)
    test_sentences = [
        "నేను నా పుస్తకం నీకు ఇస్తాను",
        "అతను అతని తల్లితో మాట్లాడతాడు",
        "మీరు మీ ఇల్లు వారికి చూపించారా?"
    ]
    
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        pronouns = engine.analyze_sentence(sentence)
        for p in pronouns:
            print(f"  {p['original']:<8} -> {p['case']:<15} ({p['english_equivalent']})")
    
    # Paradigm generation demo
    print(f"\n📚 PRONOUN PARADIGM DEMO:")
    print("=" * 100)
    for base in ["నేను", "అతను", "మీరు"]:
        paradigm = engine.get_pronoun_paradigm(base)
        print(f"\n{base} paradigm:")
        for case, form in paradigm.items():
            print(f"  {case:<15}: {form}")

if __name__ == "__main__":
    main()
