import re

class Arabic_Stemmer:
    def __init__(self):
        self.prefixes = ['ال', 'و', 'ب', 'ف', 'ل', 'ك', 'ي', 'س', 'ت', 'ن', 'أ', 'م', 'إ', 'ع', 'ح', 'ر', 'ز', 'ص', 'غ', 'ض', 'ذ', 'ش', 'ظ', 'ث', 'خ', 'ج']
        self.suffixes = ['ا','ات','ان','انا','اني','ي','ة','ت','تا','تان','تما','تن','ته','تها','تهم','تهما','تهن','تي','تين','ك','كم','كما','م','ما','ون','ن','نا','ني','ه','ها','هم','هما','هن','وا','ونا','وني','يا','يت','يتان','ين','ينا','يني']
        self.patterns = ['تفعيل','فعال','فاعل', 'فعيل', 'مفعول', 'مفعل', 'مفعيل', 'مفعلة', 'مفاعيل', 'مفاعلة','أفعال','فعالي']
        self.mapping = {'ا': '1', 'و': '2', 'ي': '3', 'م': '4','ة': '5','ت':'6'}
        self.encoded_patterns = [self.pattern_encoder(pattern) for pattern in self.patterns]

    def remove_diacritics(self, word):
        return re.sub('[\u064b-\u0652]', '', word)

    def remove_prefix(self, word, prefixes):
        for prefix in prefixes:
            if word.startswith(prefix):
                return word[len(prefix):]
        return word

    def remove_suffix(self, word, suffixes):
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def pattern_encoder(self, word):
        encoded_word = ''.join(self.mapping.get(letter, '0') for letter in word)
        if not encoded_word.startswith('4'):
            encoded_word = encoded_word.replace('4', '0', 1)
        return encoded_word

    def match_pattern(self, word, patterns):
        for pattern in patterns:
            found = True
            if len(pattern) == len(word):
                for letter in range(len(word)):
                    if pattern[letter] != '0' and pattern[letter] != word[letter]:
                        found = False
                        break
            elif len(pattern) != len(word):
                found = False
            if found == True:
                return True
        return False

    def stem(self, word):
        word = self.remove_diacritics(word)
        if self.match_pattern(self.pattern_encoder(word), self.encoded_patterns):
            return word
        temp = ''
        while len(word) > 4 and temp != word:
            temp = word
            word = self.remove_suffix(word, self.suffixes) 
            if self.match_pattern(self.pattern_encoder(word), self.encoded_patterns):
                return word
            word = temp
            word = self.remove_prefix(word, self.prefixes)
            if self.match_pattern(self.pattern_encoder(word), self.encoded_patterns):
                return word 
            word = self.remove_suffix(word, self.suffixes) 
            if self.match_pattern(self.pattern_encoder(word), self.encoded_patterns):
                return word
        return word
