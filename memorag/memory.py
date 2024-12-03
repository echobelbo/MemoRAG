from typing import Dict, Union, List, Optional
from .memorag import Model
from .retrieval import DenseRetriever
from .prompt import zh_prompts
from semantic_text_splitter import TextSplitter
import spacy
import networkx as nx
from memorag import Agent
import json
import matplotlib.pyplot as plt
import re
import community as community_louvain
import pickle
from itertools import chain
import os
# from networkx.algorithms.community import girvan_newman
# from spacy.lang.zh.examples import sentences

# api_dict = {
#   "models": ["deepseek-chat"],
#   "base_url": "https://api.deepseek.com",
#   "api_key": "sk-22e0ee691f474267b67eae0d48e962b8"
# }
# model = "deepseek-chat"
# source = "deepseek"
# # with open('./examples/report.txt', 'r', encoding='utf-8') as file:
# #     content = file.read()

# agent =  Agent(model, source, api_dict)


def girvan_newman(G: nx.Graph):
    # 复制图避免修改原始图
    G_copy = G.copy()

    # 当图中仍有多个连通组件时，继续进行边删除
    while nx.number_connected_components(G_copy) == 1:
        # 计算所有边的介数中心性
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)
        
        # 找到介数中心性最高的边并删除
        edge_to_remove = max(edge_betweenness, key=edge_betweenness.get)
        G_copy.remove_edge(*edge_to_remove)

    # 返回分割后的连通组件
    return list(nx.connected_components(G_copy))
def louvain_community_detection(G):
    # node_chunk = {node: G.nodes[node]["chunk_index"] for node in G.nodes}

    partition = community_louvain.best_partition(G, resolution=5.0)
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    return communities, partition
def get_similar_entity(model, graph, word, threshold=0.8):
    """
    根据词向量相似度判断是否为同义词，并返回归一化的词
    """
    for ent in graph.nodes:
        similarity = model(word).similarity(model(ent))
        if similarity > threshold:
            return ent
    return word

class Memory:
    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, context: str) -> str:
        pass

    def _prepare_retrieval_query(self, query, text_spans, surrogate_queries, use_memory_answer):
        retrieval_query = text_spans.split("\n") + surrogate_queries.split("\n")
        retrieval_query = [q for q in retrieval_query if len(q.split()) > 3]
        potential_answer = None
        if use_memory_answer:
            # potential_answer = self.mem_model.answer(query)
            retrieval_query.append(potential_answer)
        retrieval_query.append(query)
        return retrieval_query, potential_answer

class GraphMemory(Memory):
    def __init__(self, 
                 retriever,
                 extractor: Agent,
                 refine_model: Agent, 
                 config: Dict,
                 chunk_size:int = 256,
                 ret_model_name_or_path: str="BAAI/bge-m3",
                 ret_hit: int = 3,
                 cache_dir:Optional[str]=None):
        super().__init__(config)
        self.retriever = retriever
        self.graph = nx.Graph()
        self.nlp_model = spacy.load("zh_core_web_sm")
        self.extractor = extractor
        self.partition = {}
        self.communities = {}
        self.refine_model = refine_model
        self.chunks = None
        self.text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", chunk_size)
        # self.retriever = DenseRetriever(
        #     encoder = ret_model_name_or_path,
        #     hits=ret_hit,
        #     cache_dir=cache_dir
        # )

        

    def memorize(self, 
                 context: str,
                 load: bool = True,
                 save_dir: str = None
                 ) -> None:
        # tools and deployed model: Spacy -> 
        # API: aliyun API stc. deepseek API  -> ner / deepseek
        index = 0            
        self.chunks = self.text_splitter.chunks(context)

        node2chunk={}

        # self.retriever.remove_all()
        # self.retriever.add(self.chunks)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # self.mem_model.save(os.path.join(save_dir, "memory.bin"))
            self.retriever._index.save(os.path.join(save_dir, "index.bin"))
            with open(os.path.join(save_dir, "chunks.json"), "w") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            # self._print_stats(save_dir, context)

        if load:
            while(True):
                try:
                    with open(f'./cache/json{index}', 'r', encoding='utf-8') as f:
                        sub_graph = json.load(f)
                        for node, neighbors in sub_graph.items():
                            self.graph.add_node(node)
                            node2chunk.setdefault(node, set()).update([index])
                            for neighbor in neighbors:
                                self.graph.add_edge(node, neighbor)
                                node2chunk.setdefault(neighbor, set()).update([index])
                    index+=1
                except Exception as e:
                    if hasattr(e, 'strerror') and e.strerror == 'No such file or directory' and index !=0:
                        print("load finished")
                        break
                    else:
                        print(f"no file to load or {e}")
                        exit
        else:
            # text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", chunk_size)
            # chunk_index = 0
            for chunk in self.chunks:
                prompt = zh_prompts["ner_mem"].format(context=chunk)
                response = self.extractor.generate(prompt)
                try:
                    result = json.loads(re.search(r'\{.*\}',response[0], re.DOTALL).group(0))
                    with open(f'./cache/json{index}', 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=4)
                    for node, neighbors in result.items():
                        self.graph.add_node(node)
                        node2chunk.setdefault(node, set()).update([index])
                        for neighbor in neighbors:
                            self.graph.add_edge(node, neighbor)
                            node2chunk.setdefault(neighbor, set()).update([index])
                    print(f"{index}json, dump finished")
                except:
                    print(response)
                    print(chunk)
                index+=1
        print(self.graph)

        ''' girvan_newman'''
        # communities = girvan_newman(self.graph)
        # for i, community in enumerate(communities): 
        #     print(f"Community {i + 1}: {community}")

        '''louvain '''
        self.communities2chunk = {}
        self.communities, self.partition = louvain_community_detection(self.graph)
        for node in self.graph.nodes:
            # if 
            self.communities2chunk.setdefault(self.partition[node], set()).update(node2chunk[node])
            '''
            该数据结构为
            {
            communities1:set{chunk1,chunk2,...},
            communities2:set{chunk3,...}
            }
            '''
        print("Louvain communities:", self.communities)
        return
    
    def refine_query(self, query: str) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 
        # prompt the model to refine the query into sub-questions
        # query：北京去年总体表现怎么样？
        # output: 北京去年经济怎么样？ -> top-10 chunks 5 out 10 chunks are to the query, 
        #        北京去年民生怎么样？
        #        北京去年科技怎么样？
        #        海淀区去年怎么样？
        prompt = zh_prompts["sur"].format(question=query)
        surrogate_queries = self.refine_model.generate(prompt)
        sub_queries = surrogate_queries[0].split("\n")
        # sub_ner = [self.refine_model.generate(zh_prompts["ner_qa"].format(question=subquery)) for subquery in sub_queries]

        # TODO discuss how to refine
        # LLM refine
        # graph structure -> useful info
        # prompt
        
        # prompt = zh_prompts["ner_qa"].format(question=query)
        # query_ent = self.refine_model.generate(prompt)
        # print(query_ent)

        print(surrogate_queries)

        return sub_queries

    def match_query(self, query: str, sub_queries: List[str]) -> List[str]:
        # for each sub-query, retrieve the most relevant chunks from the long document
        # build community structure for the chunks
        # rank the chunk communities by their relevance to the input query or the combined query 
           # query + all chunks -> long sequence -> main query -> similarity
           # query + each chunks -> long sequence -> main query -> similarity
        # for each chunk community, select relevant chunks by measuring the semantic similarity between the chunks and the query
            #threshold = 0.6 reranker, matcher
        retrieval_results = self._retrieve(queries=sub_queries)
        print(retrieval_results)
        


        output = []
        return output
    
    def _retrieve(self, retrieval_query):
        topk_scores, topk_indices = self.retriever.search(queries=retrieval_query)
        topk_indices = list(chain(*[topk_index.tolist() for topk_index in topk_indices]))
        topk_indices = sorted(set([x for x in topk_indices if x > -1]))
        return [self.chunks[i].strip() for i in topk_indices]
    
    def map_answer(self, query: str, answer: str, max_length: int=1024) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 

        # recursively generate useful claims from each chunk community
        # untill fit the predefined length
        # claim: 北京去年经济稳定增长，增长率高达8%
        # claim: 北京去年民生持续改善，居民收入稳步增长
        pass

    def reduce_answer(self, query: str, answer: str) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 

        # produce the final answer from the useful claims
        pass

    def __call__(self, query: str, context: str) -> str:
        answer = ""
        return answer


class AgentMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(config)


class KVMemory(Memory):
    def __init__(self, config: Dict):
        super().__init__(config)

    def memorize(self, context: str) -> None:
        pass


if __name__ == "__main__":
    query = "什么是MemoRAG？"
    context = "MemoRAG是一种基于记忆的检索增强生成模型，它通过记忆来增强生成模型的生成能力。"
    memory = KVMemory({})
    memory.memorize(context)
    
    