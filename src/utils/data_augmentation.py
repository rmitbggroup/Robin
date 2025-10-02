import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import numpy as np
import string   

class WordLevelAugmenter:
    def __init__(self, probability=0.3):
        """
        Initialize the word-level augmenter
        Args:
            probability (float): Probability of applying augmentation to each word
        """
        self.probability = probability
        self.download_nltk_data()
        
        # Define all available intermediate languages
        self.intermediate_languages = {
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'zh': 'Chinese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'it': 'Italian',
            'nl': 'Dutch'
        }
        
        # Initialize translation models
        self.models = {}
        self.tokenizers = {}

    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('punkt')

    def initialize_translation_model(self, lang_code):
        """
        Initialize translation model for a specific language
        Args:
            lang_code (str): Language code
        """
        try:
            # Initialize English to target language model
            model_name = f'Helsinki-NLP/opus-mt-en-{lang_code}'
            self.models[f'en-{lang_code}'] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[f'en-{lang_code}'] = MarianTokenizer.from_pretrained(model_name)
            
            # Initialize target language to English model
            model_name = f'Helsinki-NLP/opus-mt-{lang_code}-en'
            self.models[f'{lang_code}-en'] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[f'{lang_code}-en'] = MarianTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading models for language {lang_code}: {e}")

    def translate(self, text, source_lang, target_lang):
        """
        Translate text between languages
        Args:
            text (str): Input text
            source_lang (str): Source language code
            target_lang (str): Target language code
        Returns:
            str: Translated text
        """
        model_key = f'{source_lang}-{target_lang}'
        if model_key not in self.models:
            self.initialize_translation_model(target_lang if source_lang == 'en' else source_lang)
        
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated

    def backtranslate_multi(self, text, num_augmentations=1):
        """
        Perform backtranslation using multiple intermediate languages
        Args:
            text (str): Input text
            num_augmentations (int): Number of augmented versions to generate
        Returns:
            list: List of augmented texts
        """
        augmented_texts = []
        available_languages = list(self.intermediate_languages.keys())
        
        for _ in range(num_augmentations):
            # Randomly select an intermediate language
            intermediate_lang = random.choice(available_languages)
            
            try:
                # Translate to intermediate language
                intermediate_text = self.translate(text, 'en', intermediate_lang)
                # Translate back to English
                back_translated = self.translate(intermediate_text, intermediate_lang, 'en')
                augmented_texts.append(back_translated)
            except Exception as e:
                print(f"Error in backtranslation using {self.intermediate_languages[intermediate_lang]}: {e}")
                augmented_texts.append(text)  # Fall back to original text
        
        return augmented_texts

    def get_synonyms_and_hypernyms(self, word):
        """
        Get synonyms and hypernyms for a word using WordNet
        Args:
            word (str): Input word
        Returns:
            list: List of candidate words (synonyms and hypernyms)
        """
        candidates = set()
        
        # Get synsets for the word
        for synset in wordnet.synsets(word):
            # Add synonyms
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    candidates.add(lemma.name())
            
            # Add hypernyms
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    candidates.add(lemma.name())
        
        return list(candidates)

    def word_removal(self, text):
        """
        Randomly remove a word from the text
        Args:
            text (str): Input text
        Returns:
            str: Augmented text with a word removed
        """
        words = word_tokenize(text)
        if len(words) <= 1:  # Don't remove if only one word
            return text
            
        # Randomly select a word to remove
        remove_idx = random.randint(0, len(words) - 1)
        words.pop(remove_idx)
        
        return ' '.join(words)

    def word_substitution(self, text):
        """
        Substitute a random word with its synonym or hypernym
        Args:
            text (str): Input text
        Returns:
            str: Augmented text with a word substituted
        """
        words = word_tokenize(text)
        if not words:
            return text
            
        # Try to find a word that has candidates
        attempts = 0
        max_attempts = len(words)  # Limit attempts to avoid infinite loop
        
        while attempts < max_attempts:
            # Randomly select a word
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            
            # Get candidate replacements
            candidates = self.get_synonyms_and_hypernyms(word)
            
            if candidates:
                # Replace with random candidate
                words[idx] = random.choice(candidates)
                break
                
            attempts += 1
        
        return ' '.join(words)

    def word_swapping(self, text):
        """
        Swap two neighboring words
        Args:
            text (str): Input text
        Returns:
            str: Augmented text with two words swapped
        """
        words = word_tokenize(text)
        if len(words) <= 1:  # Need at least 2 words to swap
            return text
            
        # Randomly select first word position (ensuring there's a next word)
        pos = random.randint(0, len(words) - 2)
        
        # Swap with next word
        words[pos], words[pos + 1] = words[pos + 1], words[pos]
        
        return ' '.join(words)

    def augment(self, text, num_augmentations=1):
        """
        Apply random augmentations to the text
        Args:
            text (str): Input text
            num_augmentations (int): Number of augmented versions to generate
        Returns:
            list: List of augmented texts
        """
        augmented_texts = []
        augmentation_methods = [
            self.word_removal,
            self.word_substitution,
            self.word_swapping
        ]
        
        for _ in range(num_augmentations):
            if random.random() < self.probability:
                # Randomly select an augmentation method
                augmentation_func = random.choice(augmentation_methods)
                augmented_text = augmentation_func(text)
                augmented_texts.append(augmented_text)
            else:
                augmented_texts.append(text)
        
        return augmented_texts

class CharacterLevelAugmenter:
    def __init__(self, probability=0.3):
        """
        Initialize the character-level augmenter
        Args:
            probability (float): Probability of applying augmentation to each character
        """
        self.probability = probability
        self.keyboard_mapping = {
            # QWERTY keyboard layout neighbors
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'srfce', 'e': 'wrsdf',
            'f': 'drtgv', 'g': 'ftyhb', 'h': 'gyujn', 'i': 'ujko', 'j': 'huikm',
            'k': 'jiol', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awdxz', 't': 'rfgy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx',
            # Numbers
            '0': '9', '1': '2', '2': '13', '3': '24', '4': '35', 
            '5': '46', '6': '57', '7': '68', '8': '79', '9': '80'
        }

    def _get_similar_char(self, char):
        """Get a similar character based on keyboard proximity"""
        if char.lower() in self.keyboard_mapping:
            similar_chars = self.keyboard_mapping[char.lower()]
            similar_char = random.choice(similar_chars)
            return similar_char if char.islower() else similar_char.upper()
        return char

    def augment_text(self, text, num_augmentations=1):
        """
        Apply character-level augmentation to text
        Args:
            text (str): Input text
            num_augmentations (int): Number of augmented versions to generate
        Returns:
            list: List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            chars = list(text)
            for i in range(len(chars)):
                if random.random() < self.probability:
                    operation = random.choice(['substitute', 'insert', 'delete', 'swap'])
                    
                    if operation == 'substitute':
                        # Substitute with similar character
                        chars[i] = self._get_similar_char(chars[i])
                    
                    elif operation == 'insert' and i < len(chars):
                        # Insert similar character
                        similar_char = self._get_similar_char(chars[i])
                        chars.insert(i + 1, similar_char)
                    
                    elif operation == 'delete' and len(chars) > 1:
                        # Delete character
                        chars.pop(i)
                        if i >= len(chars):
                            break
                    
                    elif operation == 'swap' and i < len(chars) - 1:
                        # Swap with next character
                        chars[i], chars[i + 1] = chars[i + 1], chars[i]
            
            augmented_texts.append(''.join(chars))
        
        return augmented_texts

    def augment_number(self, number, num_augmentations=1):
        augmented_numbers = []
        number_str = str(number)
        is_float = '.' in number_str
        
        for _ in range(num_augmentations):
            chars = list(number_str)
            for i in range(len(chars)):
                if chars[i] == '.':
                    continue
                    
                if random.random() < self.probability:
                    operation = random.choice(['substitute', 'swap'])
                    
                    if operation == 'substitute':
                        if chars[i] in self.keyboard_mapping:
                            chars[i] = random.choice(self.keyboard_mapping[chars[i]])
                    
                    elif operation == 'swap' and i < len(chars) - 1:
                        if chars[i + 1] != '.' and chars[i] != '.':
                            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            
            augmented_number = ''.join(chars)
            try:
                if is_float:
                    augmented_numbers.append(float(augmented_number))
                else:
                    augmented_numbers.append(int(augmented_number))
            except ValueError:
                augmented_numbers.append(number)
        
        return augmented_numbers

class DataAnalyzer:
    def __init__(self):
        """Initialize the data analyzer"""
        # Define numerical types
        self.numerical_types = [
            np.number,          # All numpy numbers
            int, float,         # Python numbers
            'int8', 'int16', 'int32', 'int64',
            'float16', 'float32', 'float64'
        ]
        
        # Define string/text types
        self.text_types = [
            str, object,        # Python strings and objects
            'string', 'object'  # Pandas string and object
        ]

    def analyze_column_type(self, series):
        """
        Analyze whether a column is numerical or text
        Args:
            series (pd.Series): Column to analyze
        Returns:
            str: 'numerical', 'text', or 'mixed'
        """
        # Check dtype first
        if pd.api.types.is_numeric_dtype(series):
            return 'numerical'
        
        # If dtype is object, analyze content
        if pd.api.types.is_object_dtype(series):
            # Remove NaN values for analysis
            non_null = series.dropna()
            
            if len(non_null) == 0:
                return 'unknown'
            
            # Check if all values are strings
            all_strings = all(isinstance(x, str) for x in non_null)
            if all_strings:
                return 'text'
            
            # Check if all values can be converted to float
            try:
                non_null.astype(float)
                return 'numerical'
            except ValueError:
                # Check if mixed type
                numeric_count = sum(isinstance(x, (int, float)) for x in non_null)
                if numeric_count > 0 and numeric_count < len(non_null):
                    return 'mixed'
                return 'text'
        
        return 'unknown'

    def analyze_dataframe(self, df):
        """
        Analyze all columns in a DataFrame
        Args:
            df (pd.DataFrame): DataFrame to analyze
        Returns:
            dict: Dictionary with column types
        """
        column_types = {}
        for column in df.columns:
            column_types[column] = {
                'type': self.analyze_column_type(df[column]),
                'unique_values': df[column].nunique(),
                'null_count': df[column].isnull().sum(),
                'sample_values': df[column].dropna().head(3).tolist()
            }
        return column_types

    def get_text_columns(self, df):
        """
        Get list of text columns in DataFrame
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            list: List of column names that contain text data
        """
        text_columns = []
        for column in df.columns:
            if self.analyze_column_type(df[column]) == 'text':
                text_columns.append(column)
        return text_columns

    def get_numerical_columns(self, df):
        """
        Get list of numerical columns in DataFrame
        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            list: List of column names that contain numerical data
        """
        numerical_columns = []
        for column in df.columns:
            if self.analyze_column_type(df[column]) == 'numerical':
                numerical_columns.append(column)
        return numerical_columns

    def print_column_analysis(self, df):
        """
        Print detailed analysis of DataFrame columns
        Args:
            df (pd.DataFrame): DataFrame to analyze
        """
        analysis = self.analyze_dataframe(df)
        print("\nColumn Analysis:")
        print("-" * 60)
        for column, info in analysis.items():
            print(f"\nColumn: {column}")
            print(f"Type: {info['type']}")
            print(f"Unique values: {info['unique_values']}")
            print(f"Null count: {info['null_count']}")
            print(f"Sample values: {info['sample_values']}")

def main():
    # Example usage
    augmenter = WordLevelAugmenter(probability=0.7)
    
    # Example text
    text = "The quick brown fox jumps over the lazy dog"
    
    print("Original text:", text)
    print("\nAugmented versions:")
    
    # Generate augmented versions using each method
    print("\n1. Word Removal:")
    print(augmenter.word_removal(text))
    
    print("\n2. Word Substitution:")
    print(augmenter.word_substitution(text))
    
    print("\n3. Word Swapping:")
    print(augmenter.word_swapping(text))
    
    print("\n4. Backtranslation with multiple languages:")
    backtranslated_texts = augmenter.backtranslate_multi(text, num_augmentations=3)
    for i, aug_text in enumerate(backtranslated_texts, 1):
        print(f"{i}. {aug_text}")
    
    print("\nRandom augmentations:")
    augmented_texts = augmenter.augment(text, num_augmentations=3)
    for i, aug_text in enumerate(augmented_texts, 1):
        print(f"{i}. {aug_text}")

    # Example usage of DataAnalyzer
    analyzer = DataAnalyzer()
    
    # Create sample DataFrame
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'age': [25, 30, 35, 28, 42],
        'description': ['Software Engineer', 'Data Scientist', 'Manager', 'Developer', 'Analyst'],
        'salary': ['50000', '60000', '75000', '55000', '65000'],  # String numbers
        'mixed': ['text', 123, 'more text', 456, 'final text']
    }
    df = pd.DataFrame(data)
    
    # Analyze the DataFrame
    analyzer.print_column_analysis(df)
    
    # Get text and numerical columns
    print("\nText columns:", analyzer.get_text_columns(df))
    print("Numerical columns:", analyzer.get_numerical_columns(df))

    # Example usage of CharacterLevelAugmenter
    print("\nCharacter-level Augmentation Examples:")
    char_augmenter = CharacterLevelAugmenter(probability=0.3)
    
    # Text augmentation
    text = "Hello World"
    print("\nOriginal text:", text)
    print("Character-level augmented texts:")
    augmented_texts = char_augmenter.augment_text(text, num_augmentations=3)
    for i, aug_text in enumerate(augmented_texts, 1):
        print(f"{i}. {aug_text}")
    
    # Number augmentation
    number = 12345.67
    print("\nOriginal number:", number)
    print("Character-level augmented numbers:")
    augmented_numbers = char_augmenter.augment_number(number, num_augmentations=3)
    for i, aug_num in enumerate(augmented_numbers, 1):
        print(f"{i}. {aug_num}")

if __name__ == "__main__":
    main() 