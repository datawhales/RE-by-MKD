import re
from transformers import BertTokenizer

class EntityMarker:
    """ Converts raw text to BERT-input ids and finds entity position.
    
    Attributes:
        tokenizer: Bert-base tokenizer.
        h_pattern: A regular expression pattern -- * h *. Used to replace head entity mention.
        t_pattern: A regular expression pattern -- ^ t ^. Used to replace tail entity mention.
        err: Records the number of sentences where we can't find head/tail entity normally.
        args: Args from command line.
    """
    def __init__(self, args=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.args = args

    def tokenize(self, raw_text, h_pos_range, t_pos_range, h_type=None, t_type=None, h_blank=False, t_blank=False):
        """ Tokenizer for 'CM', 'CT' settings.

        This function converts raw text to BERT-input ids and uses entity marker to highlight entity
        position and randomly replaces entity mention with special 'BLANK' symbol. Entity mention can
        be entity type(If h_type and t_type are not none). And this function returns ids that can be
        used as the inputs to BERT directly and entity position.

        Args:
            raw_text: A python list of tokens.
            h_pos_range: A python list of head entity position. For example, h_pos_range maybe [2, 6] which indicates
                that head entity mention = raw_text[2:6]
            t_pos_range: A python list of tail entity position.
            h_type: Head entity type. This argument is used when we use type instead of the entity mention.
            t_type: Tail entity type.
            h_blank: True when head entity mention is converted to 'BLANK' symbol else False.
            t_blank: True when tail entity mention is converted to 'BLANK' symbol else False.

        Returns:
            tokenized_input: BERT-input ids
            h_pos: Head entity marker start position.
            t_pos: Tail entity marker start position.

        Example:
            raw_text: ["Bill", "Gates", "founded", "Microsoft", "."]
            h_pos_range: [0, 2]
            t_pos_range: [3, 4]
            h_type: None
            t_type: None
            h_blank: True
            t_blank: False

            1. Replace entity mention with special pattern.
                "* h * founded ^ t ^ ."
            2. Replace pattern.
                "[CLS] [unused0] [unused4] [unused1] founded [unused2] microsoft [unused3] . [SEP]"
            3. Find the positions of entities and convert tokenized sentence to ids.
                [101, 1, 5, 2, 2631, 3, 7513, 4, 1012, 102]
                h_pos = 1
                t_pos = 5
        """
        tokens = []
        h_mention = []
        t_mention = []
        for i, token in enumerate(raw_text):
            token = token.lower()    
            if i >= h_pos_range[0] and i < h_pos_range[-1]:
                if i == h_pos_range[0]:
                    tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= t_pos_range[0] and i < t_pos_range[-1]:
                if i == t_pos_range[0]:
                    tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)

        # tokenize
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_head = self.tokenizer.tokenize(h_mention)
        tokenized_tail = self.tokenizer.tokenize(t_mention)

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        # If head entity type and tail entity type are't None, 
        # we use `CT` settings to tokenize raw text, i.e. replacing 
        # entity mention with entity type.
        if h_type != None and t_type != None:
            p_head = h_type
            p_tail = t_type

        if h_blank:
            p_text = self.h_pattern.sub("[unused0] [unused4] [unused1]", p_text)
        else:
            p_text = self.h_pattern.sub("[unused0] "+p_head+" [unused1]", p_text)
        if t_blank:
            p_text = self.t_pattern.sub("[unused2] [unused5] [unused3]", p_text)
        else:
            p_text = self.t_pattern.sub("[unused2] "+p_tail+" [unused3]", p_text)
    
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        # If h_pos_range and t_pos_range overlap, we can't find head entity or tail entity.
        try:
            h_pos = f_text.index("[unused0]")
            h_end = f_text.index("[unused1]")
        except:
            self.err += 1
            h_pos = 0
            h_end = 2
        try:
            t_pos = f_text.index("[unused2]") 
            t_end = f_text.index("[unused3]")
        except:
            self.err += 1
            t_pos = 0
            t_end = 2
        tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        
        return tokenized_input, h_pos, t_pos, h_end, t_end