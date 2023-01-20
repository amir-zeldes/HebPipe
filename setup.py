from setuptools import setup, find_packages

setup(
  name = 'hebpipe',
  packages = find_packages(),
  version = '3.0.0.2',
  description = 'A pipeline for Hebrew NLP',
  author = 'Amir Zeldes',
  author_email = 'amir.zeldes@georgetown.edu',
  package_data = {'':['README.md','LICENSE.md','requirements.txt'],'hebpipe':['lib/*','data/*','bin/*','models/models_go_here.txt','models/stanza/stanza_models_here.txt']},
   install_requires=['requests','numpy','transformers==3.5.1','torch==1.7.1','pandas','scipy','joblib','xgboost==0.81','rftokenizer','depedit','xmltodict', 'diaparser==1.1.2','flair==0.6.1','stanza','conllu','protobuf==3.20.*'],
  url = 'https://github.com/amir-zeldes/HebPipe',
  license='Apache License, Version 2.0',
  download_url = 'https://github.com/amir-zeldes/HebPipe/releases/tag/v3.0.0.2',
  keywords = ['NLP', 'Hebrew', 'segmentation', 'tokenization', 'tagging', 'parsing','morphology','POS','lemmatization'],
  classifiers = ['Programming Language :: Python',
'Programming Language :: Python :: 2',
'Programming Language :: Python :: 3',
'License :: OSI Approved :: Apache Software License',
'Operating System :: OS Independent'],
)