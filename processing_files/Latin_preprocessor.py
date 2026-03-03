import re

class LatinPreprocessor:
    
    def __init__(self, stop_words_path=None, filter_words_path=None):

        self.stop_words = stop_words_path
        self.filter_words_path = filter_words_path
        
        if self.stop_words is not None:
            with open(self.stop_words, 'r', encoding='utf-8') as f:
                self.stop_words_set = set(line.strip() for line in f)

        if self.filter_words_path is not None:
            with open(self.filter_words_path, 'r', encoding='utf-8') as f:
                self.filter_words_set = set(line.strip() for line in f)

        # Sentence splitter: . ? ! followed by whitespace
        self.sent_split_pattern = re.compile(r'([.!?\n])\s+')

        # Regex to match characters allowed in the corpus.
        # We KEEP:
        # latin characters with diacritics
        # \s            : Whitespace
        # \.,;:!?       : Specific punctuation allowed
        # Everything else (including 0-9) is replaced.
        self.allowed_chars_pattern = re.compile(r'[^A-Za-zÀ-ÿ\s\.,;:!?\'\n]')

    def clean_whitespace(self, text):
        """Collapses multiple spaces/newlines into single spaces."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?<![A-Za-z])[.,;:!?]', '', text)
        return text.strip()

    def clean_multiple_punct(self, text):
        """delete multiple consecutive dots and semicolons, preserving a single instance."""
        # drop any run of punctuation like '..', '.,', ';.'.        
        while re.search(r'([\.;,·:!?])(?:[\.;,·:!?])', text):    
            text = re.sub(r'([\.;,·:!?])(?:[\.;,·:!?])', r'', text)
        return text
    
    def aggregate_splitted_words(self, text):
        """Aggregates words that were split by line break and '-' """
        text = re.sub(r'-\s', '', text)
        text = text.replace('-','')
        return text

    def normalize(self, text):

        # 1. filter out unwanted sequences
        # seq = re.compile(r'<[^>]+>')  # matches anything between <>
        # text = seq.sub('', text)
        
        # seq = re.compile(r'\[[^\]]+\]')  # matches anything between []
        # text = seq.sub('', text)

        # seq = re.compile(r'\([^\)]+\)')  # matches anything between ()
        # text = seq.sub('', text)

        # 2. lowercase latin names if any
        # if self.filter_words_path is not None:
        #     words = text.split()
        #     lowercased_words = [word.lower() if word in self.filter_words_set else word for word in words]
        #     text = ' '.join(lowercased_words)

        # 3. Remove all uppercase letters that are not followed by lowercase letters
        # text = re.sub(r'[A-Z](?![a-z])', '', text)

        # 4. Lowercase the remaining capitals
        text = text.lower()
        # text = text.replace('j', 'i').replace('v', 'u')  # optional: normalize j/v to i/u
        text = self.allowed_chars_pattern.sub('', text)

        return text

    def drop_punctuation(self, text):
        text = re.sub(r'[,.!?:;]', '', text)
        return text
    
    def segment_sentences(self, text):
        chunks = self.sent_split_pattern.split(text)
        sentences = []
        current_sent = ""
        
        for chunk in chunks:
            if chunk in ['.', '!', '?']:
                current_sent += chunk
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""
            else:
                current_sent += chunk

        if current_sent.strip():
            sentences.append(current_sent.strip())

        return sentences

    def remove_stop_words(self, sentence):
        words = sentence.split()
        filtered_words = [word for word in words if word not in self.stop_words_set]
        return ' '.join(filtered_words)

