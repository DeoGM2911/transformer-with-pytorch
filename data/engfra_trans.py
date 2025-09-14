# engfra_trans.py
#
# The English-French translation dataset (from the Tatoeba dataset).
#
# @author: Dung Tran
# @date: September 10, 2025

import torch
import regex as re
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle


class MTEngFra(Dataset):
    """
    The English-French translation dataset (from the Tatoeba dataset).
    Each sample is a tuple of (source_sentence, target_sentence).
    """
    def __init__(self, data_path, num_steps, num_train, num_val):
        """
        Initialize the dataset.

        Parameters:
        ----------
        data_path : str
            The directory path where the dataset will be stored.
        transform : callable, optional
            A function/transform to apply to the data.
        """
        super(MTEngFra, self).__init__()
        self.data = []
        self.data_path = data_path
        
        # Download and get the path to the data file
        self.file_path = self._download()
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        
        # Load the data
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(self.file_path, None, None)
    
    def __len__(self):
        return len(self.arrays[0])
    
    def __getitem__(self, idx):
        return self.arrays[0][idx], self.arrays[1][idx], self.arrays[2][idx], self.arrays[3][idx]
    
    def data_loader(self, batch_size, train=True):
        # Take a slice of all the arrays
        data_batch = [None] * len(self.arrays)
        if train:
            start = 0
            end = self.num_train
        else:
            start = self.num_train
            end = self.num_train + self.num_val
        
        for i in range(len(self.arrays)):
            # Skip if enc valid length is none
            if self.arrays[i] is None:
                continue
            
            data_batch[i] = shuffle(self.arrays[i][start:end])
        
        # Yield batch_size amount of samples
        num_batch = len(data_batch[0]) // batch_size
        for i in range(num_batch):
            start, end = i * batch_size, min((i + 1) * batch_size, len(data_batch[0]))
            if data_batch[3] is None:
                yield data_batch[0][start:end], data_batch[1][start:end], data_batch[2][start:end], None
            yield data_batch[0][start:end], data_batch[1][start:end], data_batch[2][start:end], data_batch[3][start:end]

    
    def _download(self):
        """
        Download the English-French dataset from the given URL and extract it to data_path.
        """
        import os
        import urllib.request
        import zipfile
        
        url = "http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip"
        
        # Create the data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
        
        # Check if the directory already contains the extracted data
        extracted_dir = os.path.join(self.data_path, "fra-eng")
        if os.path.exists(extracted_dir):
            print("Dataset already downloaded and extracted.")
        else:
            
            # Download the zip file
            zip_path = os.path.join(self.data_path, "fra-eng.zip")
            if not os.path.exists(zip_path):
                print(f"Downloading dataset from {url}...")
                urllib.request.urlretrieve(url, zip_path)
                print("Download completed!")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print("Extracting files...")
                zip_ref.extractall(self.data_path)
                print("Extraction completed!")
            
            # Remove the zip file after extraction (optional)
            os.remove(zip_path)
            print("Removed zip file.")
            
        # Read the file and return the raw texts
        with open(os.path.join(self.data_path, "fra-eng", "fra.txt"), 'r', encoding='utf-8') as f:
            raw_texts = f.read()
        return raw_texts
        
    
    def _preprocess(self, text):
        # Replace non-breaking spaces with regular spaces
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        
        # Separate words and punctuations
        text = re.sub(r'([!.,?])', r' \1 ', text)
        
        # Remove duplicated vanila white spaces
        text = re.sub(r' +', ' ', text)
        return text.lower()  # Lower case for simplicity

    def _tokenize(self, text, max_examples=None):
        # Simple whitespace tokenizer
        # grab the source and target sentences
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples is not None and i > max_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                # Skip empty tokens
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt

    def _build_arrays(self, raw_texts, src_vocab, tgt_vocab):
        # Build the arrays for the source and target texts
        def _build_array(tokens, vocab, is_tgt=False, min_freq=2):
            # Add padding token or truncate to the desired length
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq))
            )
            tokens = [pad_or_trim(sent, self.num_steps) for sent in tokens]
            if is_tgt:
                tokens = [['<bos>'] + sent for sent in tokens]
            if vocab is None:
                # Build the vocabulary
                vocab = {}
                i = 0
                freq = {}
                for sent in tokens:
                    for token in sent:
                        freq[token] = freq.get(token, 0) + 1
                
                for token in freq:
                    if freq[token] >= min_freq:
                        vocab[token] = i
                        i += 1

                vocab['<unk>'] = len(vocab)
            
            # Convert tokens to indices
            # Mark all tokens with low freq as <unk>
            for sent in tokens:
                for i, token in enumerate(sent):
                    if token not in vocab:
                        sent[i] = '<unk>'
            
            array = torch.tensor(
                [[vocab[token] for token in sent if token in vocab] for sent in tokens],
                dtype=torch.int32
            )
            valid_lens = (array != vocab["<pad>"]).type(torch.int32).sum(1)
            return array, valid_lens, vocab
        
        src, tgt = self._tokenize(self._preprocess(raw_texts), self.num_train + self.num_val)
        src_array, src_valid_lens, src_vocab = _build_array(src, src_vocab)
        tgt_array, _, tgt_vocab = _build_array(tgt, tgt_vocab, True)
        # Note the the second entry of the first item is the decoder expected input, the third is decoder expected output
        return ((src_array, tgt_array[:,:-1], tgt_array[:,1:], src_valid_lens),
            src_vocab, tgt_vocab)  
    
    def build(self, src_sentences, tgt_sentences):
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
            src_sentences, tgt_sentences)])
        arrays, _, _ = self._build_arrays(
            raw_text, self.src_vocab, self.tgt_vocab)
        return arrays


# Unit test
if __name__ == "__main__":
    dataset = MTEngFra("data", num_steps=5, num_train=100, num_val=10)
    loader = DataLoader(dataset, 16, True)
    for batch in loader:
        print(batch)
        break