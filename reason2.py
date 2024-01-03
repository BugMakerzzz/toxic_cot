import os
import json
import requests
import random
from qwikidata.sparql import return_sparql_query_results
# import wikipedia

os.environ["http_proxy"] = "http://Sept:20001228@127.0.0.1:14396"
os.environ["https_proxy"] = "http://Sept:20001228@127.0.0.1:14396"
os.environ["all_proxy"] = "http://Sept:20001228@127.0.0.1:14396"
import time


def query_triples(relation_id, limit_num):
    sparql_query = """
    SELECT ?headEntity ?tailEntity
    WHERE {{
      ?headEntity wdt:{} ?tailEntity.
      FILTER(STRSTARTS(STR(?headEntity), "http://www.wikidata.org/entity/Q"))
      FILTER(STRSTARTS(STR(?tailEntity), "http://www.wikidata.org/entity/Q"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {}
    """.format(relation_id, limit_num)
    while True:
        try:
            results = return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results'][
                'bindings']
            break
        except:
            time.sleep(0.5)
    return results


def query_ntail(entity_id, relation_id):
    sparql_query = """
    SELECT ?tailEntity
    WHERE {{
      wd:{} wdt:{} ?tailEntity.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """.format(entity_id, relation_id)
    try:
        results = return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']
    except:
        results = []
    return results


def query_title(entity_id):
    sparql_query = """
    SELECT ?articleTitle
    WHERE {{
      wd:{} rdfs:label ?articleTitle.
      FILTER(LANG(?articleTitle) = "en")
    }}
    """.format(entity_id)
    try:
        results = return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']
    except:
        results = []
    return results


def query_alias(entity_id):
    sparql_query = """
    SELECT ?alias
    WHERE {{
      wd:{} skos:altLabel ?alias.
      FILTER(LANG(?alias) = "en")
    }}
    """.format(entity_id)
    try:
        alias = return_sparql_query_results(sparql_query, "http://query.wikidata.org/sparql")['results']['bindings']
    except:
        time.sleep(0.5)

    results = []
    for a in alias:
        results.append(a['alias']['value'])
    return results


def query_pop(title):
    headers = {'User-Agent': 'zhuoran.jin@nlpr.ia.ac.cn'}
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{}/monthly/2023010100/2023013100'
    try:
        resp = requests.get(url.format(title), headers=headers)
    except:
        return -1
    if 'items' not in resp.json():
        return -1
    views = resp.json()['items'][0]['views']
    return views
step1 = {
    'P50': 'Person A is the author of {}.',
    'P57': 'Person A is the director of {}.',
    'P58': 'Person A is the screenwriter of {}.',
    'P86': 'Person A is the composer of {}.',
    'P169': 'Person A is the chief executive officer of {}.',
    'P170': 'Person A is the creator of {}.',
}

step2 = {
    'P27': 'Person A is the country of citizenship of Country B.',
}

step3 = {
    'P17': 'In which country is Country B located?',
    'P30': 'Which continent is Country B located on?',
    'P36': 'What is the capital of Country B?',
    'P37': 'What is the official language of Country B?',
}


if __name__ == '__main__':
    limit_num = 500000
    sample_num = 2000
    for r1, temp1 in step1.items():
        triples1 = query_triples(r1, limit_num)
        random.shuffle(triples1)
        triples1 = triples1[:sample_num]
        for triple1 in triples1:
            h1 = triple1['headEntity']['value'].split('/')[-1]
            t1 = triple1['tailEntity']['value'].split('/')[-1]
            h1_title = query_title(h1)
            t1_title = query_title(t1)
            if len(h1_title) == 0 or len(t1_title) == 0:
                continue
            h1_title = h1_title[0]['articleTitle']['value']
            t1_title = t1_title[0]['articleTitle']['value']
            t1_pop = query_pop(t1_title)
            if t1_pop < 1000:
                continue
            for r2, temp2 in step2.items():
                triples2 = query_ntail(t1, r2)
                if len(triples2) > 1:
                    continue
                for triple2 in triples2:
                    t2 = triple2['tailEntity']['value'].split('/')[-1]
                    t2_title = query_title(t2)
                    if len(t2_title) == 0:
                        continue
                    t2_title = t2_title[0]['articleTitle']['value']
                    for r3, temp3 in step3.items():
                        triples3 = query_ntail(t2, r3)
                        if len(triples3) > 1:
                            continue
                        for triple3 in triples3:
                            t3 = triple3['tailEntity']['value'].split('/')[-1]
                            t3_title = query_title(t3)
                            if len(t3_title) == 0:
                                continue
                            t3_title = t3_title[0]['articleTitle']['value']

                            output = {
                                'e1': t1,
                                'e1_title': t1_title,
                                'e2': h1,
                                'e2_title': h1_title,
                                'e3': t2,
                                'e3_title': t2_title,
                                'e4': t3,
                                'e4_title': t3_title,
                                'r1': r1,
                                'r2': r2,
                                'r3': r3,
                                'question': temp1.format(h1_title) + ' ' + temp2.format(t2_title) + ' ' + temp3,
                                'answer': t3_title,
                            }
                            print(temp1.format(h1_title) + ' ' + temp2.format(t2_title) + ' ' + temp3, t3_title)
                            with open('./data_3.json', 'a') as f:
                                f.write(json.dumps(output) + '\n')

