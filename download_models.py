# #模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('BAAI/bge-reranker-large',cache_dir="./models")

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('sungw111/text2vec-base-chinese-sentence',cache_dir='./models')