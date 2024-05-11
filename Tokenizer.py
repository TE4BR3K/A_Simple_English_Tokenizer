import os
import pickle
class tokenizer:
    train_dataset=[]
    vocab_size=0
    size_temp=0
    pre_tokenized_corpus=[] #list[list[(str,(int,int))]]
    word2count=dict()
    vocabs=set()
    token=dict()
    word2splits=dict()
    word2count=dict()
    pair2count = dict()
    merge_rules = []
    word_end='\\'
    seq_end='#'
    seq_start='$'
    alpha=[a for a in 'abcdefghijklmnopqrstuvwxyz.,?\'\"|:;!#@$%&()[]']+[word_end,seq_end,seq_start]
    
    def __init__(self,vocab_size=50) -> None:
        self.vocab_size=vocab_size
        pass
    def str_normalize(self,s)->str:
        return ''.join(char for char in s if ord(char) < 128).replace('<','(').replace('>',')').replace('.',' .').replace(',',' ,').replace('?',' ?').replace('!',' !')
    def pre_tokenize_split(self,source:str)->list[(str,tuple[int,int])]:
        res=[]
        start_index=end_index=0
        for word in self.str_normalize(source).split():
            end_index=start_index+len(word)
            res.append((word+self.word_end,(start_index,end_index)))
            start_index=end_index
        return res
    def pre_tokenize(self)->None:
        self.pre_tokenized_corpus=[self.pre_tokenize_split(spl) for spl in self.train_dataset]
    
    def load_train_dataset(self,train_dataset:set[str])->None:
        self.train_dataset=[strs for strs in train_dataset]
        
    def print_train_dataset(self)->None:
        print(self.train_dataset)
        print(self.pre_tokenized_corpus)
        print(self.word2count)
        print(self.vocabs)
        print(self.word2splits)
        print(self.pair2count)
    def print_vocab(self):
        print(self.vocabs)
        
    def word_count(self):
        for split_text in self.pre_tokenized_corpus:
            for word, _ in split_text:
                self.word2count[word]=self.word2count[word]+1 if word in self.word2count else 1
                
    
    def establish_vocab_set(self):
        vocab_set = set()
        for word in self.word2count:
            vocab_set.update(list(word))
        self.vocabs = vocab_set
    
    def establish_word_splits(self):
        self.word2splits = {word: [c for c in word] for word in self.word2count}
    
    def compute_pair2score(self):
        for word, word_count in self.word2count.items():
            split = self.word2splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                self.pair2count[pair] =self.pair2count[pair]+ word_count if pair in self.pair2count else word_count
                
    def compute_most_score_pair(self):
        best_pair = None
        max_freq = None
        for pair, freq in self.pair2count.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        return best_pair
    
    def make_merge_rule(self):
        best_pair = self.compute_most_score_pair()
        self.vocabs.add(best_pair[0] + best_pair[1])
        self.merge_rules.append(best_pair)
        
    def _merge_a_pair(self,a, b):
        new_word2splits = dict()
        for word, split in self.word2splits.items():
            if len(split) == 1:
                new_word2splits[word] = split
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            new_word2splits[word] = split
        return new_word2splits
    
    def merge_pairs(self):
        size_temp=0
        gap=10000
        i=0
        while len(self.vocabs) < self.vocab_size:
            self.compute_pair2score()
            best_pair = self.compute_most_score_pair()
            self.vocabs.add(best_pair[0] + best_pair[1])
            self.merge_rules.append(best_pair)
            self.word2splits = self._merge_a_pair(best_pair[0], best_pair[1])
            if size_temp==len(self.vocabs):
                i+=1
                if i==gap:
                    break
            else :
                i=0
            print(size_temp)
            size_temp=len(self.vocabs)
            
    def train(t):
        t.pre_tokenize()
        t.word_count()
        t.establish_vocab_set()
        t.establish_word_splits()
        t.compute_pair2score()
        t.make_merge_rule()
        t.merge_pairs()
        t.vocabs_list=list(t.vocabs)
        with open('merge_rules.pickle','wb') as file:
            pickle.dump(t.merge_rules,file)
        with open('vocabs_list.pickle','wb') as file:
            pickle.dump(t.vocabs_list,file)
            
    def read_trained_data(self):
        with open('merge_rules.pickle','rb') as file:
            self.merge_rules=pickle.load(file)
        with open('vocabs_list.pickle','rb') as file:
            self.vocabs_list=pickle.load(file)
            self.vocabs_list=list(self.vocabs_list)
            
    def tokenize(self, text: str) -> list[str]:
        words = [word for word, _ in self.pre_tokenize_split(text)]
        splits = [[c for c in word] for word in words]
        for merge_rule in self.merge_rules:
            for index, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == merge_rule[0] and split[i + 1] == merge_rule[1]:
                        split = split[:i] + ["".join(merge_rule)] + split[i + 2:]
                    else:
                        i += 1
                splits[index] = split
        return sum(splits, [])
    
    def encode(self,tokens)->list[int]:
        return [self.vocabs_list.index(token) for token in tokens]
    
    def decode(self,sequence)->list[str]:
        return [self.vocabs_list[idx] for idx in sequence]
            
    

if __name__=='__main__':
    t=tokenizer(vocab_size=2500)
    with open('train_data.txt','r') as file:
        t.load_train_dataset(''.join(char for char in file.read() if ord(char) < 128).split('\n'))
    t.train()
    #t.print_vocab()
    #t.read_trained_data()
    sample_str="I am the storm that is approaching!Provoking black cloud in isolation!I am reclaimer of my name!Born in flame,i have been blessed,my family crest is a demon of death!"
    print(t.decode(t.encode(t.tokenize(sample_str))))