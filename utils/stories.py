import openpyxl

REL_EXCEL_FILE_PATH = '../data/source/4000-Stories-with-sentiment-analysis.xlsx'
SHEET_NAME = 'Sheet1'

class Stories:
    """
    Treats xlsx as single source of truth for stories and provides functionality for interacting with it
    """

    def __init__(self) -> None:
        # load xlsx during initialization only
        self._stories_xlsx = openpyxl.load_workbook(REL_EXCEL_FILE_PATH, data_only=True)
        self._sheet = self._stories_xlsx[SHEET_NAME]


    def get_title_from_id(self, id: int):
        return self._sheet['D' + str(id + 2)].value

    def get_author_from_id(self, id: int):
        return self._sheet['F' + str(id + 2)].value

    def get_length_from_id(self, id: int):
        return self._sheet['C' + str(id + 2)].value

    def get_word_count_from_id(self, id: int):
        return self._sheet['M' + str(id + 2)].value

    def get_unique_words_from_id(self, id: int):
        return self._sheet['P' + str(id + 2)].value

    def get_sent_length_from_id(self, id: int):
        return self._sheet['N' + str(id + 2)].value


# example usage #
#test = Stories()
#print(test.get_title_from_id(0))
#print(test.get_author_from_id(0))
#print(test.get_length_from_id(0))
