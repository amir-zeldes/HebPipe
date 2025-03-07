from setuptools import setup, find_packages

setup(
  name = 'hebpipe',
  packages = find_packages(),
  version = '4.0.0.0',
  description = 'A pipeline for Hebrew NLP',
  author = 'Amir Zeldes',
  author_email = 'amir.zeldes@georgetown.edu',
  package_data = {'':['README.md','LICENSE.md','requirements.txt'],'hebpipe':['lib/*','data/*','bin/*','models/models_go_here.txt','models/stanza/stanza_models_here.txt']},
   install_requires=['requests','numpy','gensim==4.3.2','transformers==4.35.2','torch==2.1.0','pandas==2.1.2','scipy','joblib==1.3.2','xgboost==2.0.3','rftokenizer>=2.2.0','depedit>=3.3.1','xmltodict==0.13.0', 'diaparser==1.1.2','flair==0.13.0','stanza==1.7.0','conllu==4.5.3','protobuf==4.23.4'],
  url = 'https://github.com/amir-zeldes/HebPipe',
  license='Apache License, Version 2.0',
  download_url = 'https://github.com/amir-zeldes/HebPipe/releases/tag/v4.0.0.0',
  keywords = ['NLP', 'Hebrew', 'segmentation', 'tokenization', 'tagging', 'parsing','morphology','POS','lemmatization'],
  classifiers = ['Programming Language :: Python',
'Programming Language :: Python :: 2',
'Programming Language :: Python :: 3',
'License :: OSI Approved :: Apache Software License',
'Operating System :: OS Independent'],
)