"""Truncate story around maximum length by selecting sentences using tf-idf.

"""
import sys
sys.path.append("../")
from utils.utils import load_object_from_pkl



def main():
    
    fname = "tfidf_dict.pkl"

    tf_idf_dict = load_object_from_pkl(fname)
    print(len(tf_idf_dict))
    print(tf_idf_dict)


if __name__ == "__main__":
    main()
