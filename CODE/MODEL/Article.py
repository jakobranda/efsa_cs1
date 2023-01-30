class Article:
    """An article begin read in from a json file."""

    def __init__(self, file_name):
        """ 
        Constructs an article from a json file 

            Args:
                file_name: filename of json file
        """
        self._file_name = file_name
        with open(file_name) as json_file:
            self._article = json.load(json_file)
            
    def _line_nr_bracket(self, previous_line_start, next_line_start):
        start_line_nr = None
        for line_nr, line in enumerate(self._article[0]['lines']):
            if any(map(lambda x: line['text'].startswith(x), previous_line_start)):
                start_line_nr = line_nr + 1
            if any(map(lambda x: line['text'].startswith(x), next_line_start)) and start_line_nr is not None:
                end_line_nr = line_nr
                return start_line_nr, end_line_nr
        raise ValueError(f'object not found, markers: "{previous_line_start}" -> "{next_line_start}"')
    
    @property
    def title(self):
        try:
            start_line_nr, end_line_nr = self._line_nr_bracket(['doi'],
                                                               ['EFSA Panel',
                                                                'European Food Safety Authority',
                                                                'EFSA (European Food',
                                                                'EFSA Food',
                                                                'EFSA Scientific',
                                                                'EFSA BIOHAZ',
                                                               ])
            return ' '.join(map(itemgetter('text'),
                                self._article[0]['lines'][start_line_nr:end_line_nr]))
        except ValueError:
            raise ValueError('no title found')
    
    @property
    def panel(self):
        for line in self._article[0]['lines']:
            if line['text'].startswith('EFSA Panel'):
                return line['text']
        return None
        
    @property
    def authors(self):
        try:
            start_line_nr, end_line_nr = self._line_nr_bracket(['EFSA Panel',
                                                                'European Food Safety Authority',
                                                                'EFSA (European Food',
                                                                'EFSA Food',
                                                                'EFSA Scientific',
                                                                'EFSA BIOHAZ',
                                                               ],
                                                               ['Abstract'])
            author_str = ' '.join(map(itemgetter('text'),
                                  self._article[0]['lines'][start_line_nr:end_line_nr]))
            return author_str.replace(' and ', ', ').split(', ')
        except ValueError:
            raise ValueError('no authors found')
            
    @property
    def abstract(self):
        try:
            start_line_nr, end_line_nr = self._line_nr_bracket(['Abstract'], ['Keywords:', 'Requestor'])
            return ' '.join(map(itemgetter('text'),
                                self._article[0]['lines'][start_line_nr:end_line_nr]))
        except ValueError:
            raise ValueError('no abstract found')
            
    @property
    def keywords(self):
        try:
            start_line_nr, end_line_nr = self._line_nr_bracket(['Keywords:'],
                                                               ['Requestor',
                                                                '*',
                                                                'www.efsa'])
            start_line_nr -= 1  # keywords are listed on the line starts with Keywords
            keyword_str = ' '.join(map(itemgetter('text'),
                                   self._article[0]['lines'][start_line_nr:end_line_nr]))
            return keyword_str.replace('Keywords: ', '').split(', ')
        except ValueError:
            warnings.warn(f'no keywords found for {self._file_name}')
            return []