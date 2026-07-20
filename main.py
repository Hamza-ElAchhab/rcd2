import os
import sys
import re




re.sub(r"([A-Z][a-z])", r"\1 \2", text)



def chunking_data(repo):

    if not os.path.exists(repo):
        print("Error")
        sys.exit(1)
    
    for files, folder, root in os.walk(repo):
        print("files %s" % files)









chunking_data()

