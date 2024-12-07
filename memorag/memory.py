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
def louvain_community_detection(G, resolution):
    # node_chunk = {node: G.nodes[node]["chunk_index"] for node in G.nodes}

    partition = community_louvain.best_partition(G, resolution=resolution)
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    return communities, partition
def get_similar_entity(model, graph, word, threshold=0.7):
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
                 ret_hit: int = 10,
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
        self.topk = ret_hit
        self.text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", chunk_size)
        self.retriever = DenseRetriever(
            encoder = ret_model_name_or_path,
            hits=ret_hit,
            cache_dir=cache_dir
        )
        self.budget = 0
        

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
        self.chunk2node={}

        self.retriever.remove_all()
        self.retriever.add(self.chunks)

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
                            self.chunk2node.setdefault(index, set()).add(node)
                            node2chunk.setdefault(node, set()).update([index])
                            for neighbor in neighbors:
                                self.graph.add_edge(node, neighbor)
                                node2chunk.setdefault(neighbor, set()).update([index])
                                self.chunk2node.setdefault(index, set()).add(neighbor)
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
                        self.chunk2node.setdefault(index, set()).add(node)
                        for neighbor in neighbors:
                            self.graph.add_edge(node, neighbor)
                            self.chunk2node.setdefault(index, set()).add(node)
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
        self.communities, self.partition = louvain_community_detection(self.graph, resolution=5)
        # self.chunk2node={}
        # for node, chunks in node2chunk.items():
        #     for chunk in chunks:
        #     # 将 chunk 映射到对应的节点
        #         self.chunk2node.setdefault(chunk, set()).add(node)

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
        self.chunks2communities = {}

        for community, chunks in self.communities2chunk.items():
            for chunk in chunks:
                self.chunks2communities.setdefault(chunk, set()).add(community)

        print("Louvain communities:", self.communities)
        return
    
    def refine_query(self, query: str, graph_refine:bool=False) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 
        # prompt the model to refine the query into sub-questions
        # query：北京去年总体表现怎么样？
        # output: 北京去年经济怎么样？ -> top-10 chunks 5 out 10 chunks are to the query, 
        #        北京去年民生怎么样？
        #        北京去年科技怎么样？
        #        海淀区去年怎么样？
        prompt = zh_prompts["sur"].format(question=query)
        surrogate_queries = self.refine_model.generate(prompt)[0]
        sub_queries = surrogate_queries.split("\n")
        if graph_refine:
            sub_ner = [self.refine_model.generate(zh_prompts["ner_qa"].format(question=subquery))[0] for subquery in sub_queries]
            nlp = spacy.load("zh_core_web_md")
            # ent2node={}
            refine_sub_queries = []
            for index, ent_list in enumerate(sub_ner):
                key_words = set()
                ent_list = ent_list.split(", ")
                for ent in ent_list:
                    # if ent in ent2node.keys():
                    #     continue
                    word1=nlp(ent)
                    for node in self.graph.nodes:
                        word2=nlp(node)
                        if word1.similarity(word2) >= 0.7:
                            # ent2node.setdefault(ent, []).append(node)
                            key_words.update(self.graph.neighbors(node))
                            key_words.add(node)
                key_words=",".join(key_words)
                prompt=zh_prompts["keyword_refine"].format(query=sub_queries[index], key_words=key_words)
                refine_sub_queries.append(self.refine_model.generate(prompt)[0])
            print(refine_sub_queries)
            return refine_sub_queries
        return sub_queries



    def assess_relevance(self, chunks, query):
        """
        使用 LLM 评估文本块的相关性。
        """
        relevant_chunks = []
        # relevant_score = []
        for chunk in chunks:
            relevance_score = self.llm_assessor(self.chunks[chunk], query)  # 假设llm_assessor返回相关性分数
            # print(relevance_score)
            if relevance_score > 0.4:  # 假设0.4为相关性阈值
                relevant_chunks.append((chunk, relevance_score))
        self.budget -= 1
        return relevant_chunks
    def llm_assessor(self, chunk, query):
        prompt = zh_prompts["rel"].format(query=query, chunk=chunk)
        result=self.refine_model.generate(prompt=prompt)[0]
        try:
            result = float(result)
            return result
        except ValueError:
            print(f"无法将回答{result}转化为浮点数")
            return 0
        


    def match_query(self, query: str, sub_queries: List[str], budget:int = 20, zero_limits:int = 3) -> List[int]:
        # for each sub-query, retrieve the most relevant chunks from the long document
        # build community structure for the chunks
        # rank the chunk communities by their relevance to the input query or the combined query 
           # query + all chunks -> long sequence -> main query -> similarity
           # query + each chunks -> long sequence -> main query -> similarity
        # for each chunk community, select relevant chunks by measuring the semantic similarity between the chunks and the query
            #threshold = 0.6 reranker, matcher
        # retrieval_results = self._retrieve(retrieval_query=sub_queries)
        final_chunk=set()
        topk_scores, topk_indices = self.retriever.search(queries=sub_queries)
        
        for query_indice in range(len(sub_queries)):
            self.budget = budget
            community_scores = {}
            for i, chunk_indice in enumerate(topk_indices[query_indice]):
                for community in self.chunks2communities[chunk_indice]:
                    community_scores.setdefault(community, 0)
                    community_scores[community] += topk_scores[query_indice][i]
            ranked_community = [key for key, value in sorted(community_scores.items(), key=lambda item: item[1], reverse=True)]
            
            visited_chunks = set()
            visited_communities = set()
            relevant_chunks_score_prepare = []
            while ranked_community and self.budget > 0:
                no_relevance_count = 0
                rel_chunk_info = []
                print(f"准备访问社区{ranked_community}")
                for community in ranked_community:
                    if community in visited_communities:
                        ranked_community.remove(community)
                        continue
                    visited_communities.add(community)

                    untested_chunks = [c for c in self.communities2chunk[community] if c not in visited_chunks]
                    visited_chunks.update(untested_chunks)
                    relevant_chunks_score= self.assess_relevance(untested_chunks, sub_queries[query_indice])

                    if relevant_chunks_score == []:
                        no_relevance_count += 1
                    else:
                        relevant_chunks_score_prepare.extend(relevant_chunks_score)
                        # rel_chunk_info.append(relevant_chunks)
                        no_relevance_count = 0
                    
                    if no_relevance_count >= zero_limits:
                        '''进行一次游走'''
                        print("进行一次游走")
                        new_community = set()
                        new_community_scores = {}
                        relevant_chunks_prepare = [sublist[0] for sublist in relevant_chunks_score_prepare]
                        relevant_scores_prepare = [sublist[1] for sublist in relevant_chunks_score_prepare]
                        for i, chunk in enumerate(relevant_chunks_prepare):
                            new_community.update(self.chunks2communities[chunk])
                            for community in self.chunks2communities[chunk]:
                                new_community_scores.setdefault(community, 0)
                                new_community_scores[community] += relevant_scores_prepare[i]
                        new_community = sorted(list(new_community - visited_communities), reverse=True)
                        ranked_community = list(new_community)
                        print(f"已访问社区{visited_communities}")
                        break
            final_chunk.update([sublist[0] for sublist in relevant_chunks_score_prepare])
            print(f"查询{query_indice}查询完毕，找到了如下chunks{relevant_chunks_score_prepare}")
        # print("final_chunk")            
        return list(final_chunk)
    
    def _retrieve(self, retrieval_query):
        topk_scores, topk_indices = self.retriever.search(queries=retrieval_query)
        topk_indices = list(chain(*[topk_index.tolist() for topk_index in topk_indices]))
        topk_indices = sorted(set([x for x in topk_indices if x > -1]))
        return topk_indices
    
    def map_answer(self, query: str, answer: str, max_length: int=1024, load: bool = True) -> str:
        # open-sourced: qwen 2.5 3B, llama3.2 3B 
        # API: gpt3.5 API stc. deepseek API 

        # recursively generate useful claims from each chunk community
        # untill fit the predefined length
        # claim: 北京去年经济稳定增长，增长率高达8%
        # claim: 北京去年民生持续改善，居民收入稳步增长
        sub_graph = nx.Graph()
        node2chunk={}
        if load:
            for index in answer:
                 with open(f'./cache/json{index}', 'r', encoding='utf-8') as f:
                        graph = json.load(f)
                        for node, neighbors in graph.items():
                            sub_graph.add_node(node)
                            node2chunk.setdefault(node, set()).update([index])
                            for neighbor in neighbors:
                                sub_graph.add_edge(node, neighbor)
                                node2chunk.setdefault(neighbor, set()).update([index])
        else:
            for index in answer:
                chunk = self.chunks[index]
                prompt = zh_prompts["ner_mem"].format(context=chunk)
                response = self.extractor.generate(prompt)
                result = json.loads(re.search(r'\{.*\}',response[0], re.DOTALL).group(0))
                for node, neighbors in result.items():
                    sub_graph.add_node(node)
                    node2chunk.setdefault(node, set()).update([index])
                    for neighbor in neighbors:
                        sub_graph.add_edge(node, neighbor)
                        node2chunk.setdefault(neighbor, set()).update([index])
        
        communities2chunk = {}
        sub_comm, sub_partition = louvain_community_detection(sub_graph, resolution=1)
        claim_list = []
        for node in sub_graph.nodes:
            # if 
            communities2chunk.setdefault(sub_partition[node], set()).update(node2chunk[node])

        community_graph = nx.Graph()
        '''根据communities2chunk建立一个子图，强连通分量聚为一类'''
        community_graph.add_nodes_from(answer)
        for community, chunks in  communities2chunk.items():
            chunks = list(chunks)
            for i in range(len(chunks)):
                # community_graph.add_node()
                for j in range(i + 1, len(chunks)):
                    community_graph.add_edge(chunks[i], chunks[j])

        connected_components = list(nx.connected_components(community_graph))
        # chunk_to_cluster = {}
        # for cluster_id, component in enumerate(connected_components):
        #     for chunk in component:
        #         chunk_to_cluster[chunk] = cluster_id

        for component in connected_components:
            chunks = list(component)
            group_text = "".join([self.chunks[chunk] + "\n" for chunk in chunks])
            prompt = zh_prompts["claim_gen"].format(query = query, group_text = group_text)
            claim = self.refine_model.generate(prompt)[0]
            claim_list.extend(claim.split("\n"))



        # for community, chunks in communities2chunk.items():
        #     group_text = ""
        #     for chunk in chunks:
        #         group_text.join(self.chunks[chunk]+"\n")
        #     prompt = zh_prompts["claim_gen"].format(query = query, context = group_text)
        #     claim = self.refine_model.generate(prompt)[0]
        #     claim_list.extend(claim.split("\n"))


        scored_claims = [(claim, self.llm_assessor(chunk = claim, query = query)) for claim in claim_list]
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        selected_claims = []
        current_length = 0
        
        for claim, _ in scored_claims:
            if current_length + len(claim) + 1 <= max_length:  # +1 为换行符
                selected_claims.append(claim)
                current_length += len(claim) + 1
            else:
                break
        
        return selected_claims









        

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
    
    