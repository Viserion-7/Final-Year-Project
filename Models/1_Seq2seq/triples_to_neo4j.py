# triples_to_neo4j.py
# Convert triples CSV -> nodes.csv and edges.csv for Neo4j import, plus import.cypher script.
#
# Input triples CSV columns: subject, subject_type, predicate, object, object_type, sentence_id, source_file

import argparse
import pandas as pd
from pathlib import Path
import re
import json

def normalize_name(s):
    return re.sub(r'\s+',' ', str(s).strip()).lower()

def main(args):
    triples = pd.read_csv(args.triples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # canonicalize node names
    nodes = {}
    node_rows = []
    edges = []
    node_id = 1
    for _, row in triples.iterrows():
        sname = normalize_name(row['subject'])
        oname = normalize_name(row['object'])
        s_label = row.get('subject_type', 'Entity')
        o_label = row.get('object_type', 'Entity')
        for name, label in [(sname, s_label), (oname, o_label)]:
            if name not in nodes:
                nid = f"N{node_id}"
                nodes[name] = {'id': nid, 'name': name, 'label': label}
                node_rows.append({'id:ID': nid, 'name': name, 'label': label})
                node_id += 1
        # create edge
        edges.append({
            ':START_ID': nodes[sname]['id'],
            ':END_ID': nodes[oname]['id'],
            'relationship': row.get('predicate', 'RELATED_TO'),
            'source': row.get('source_file', ''),
            'sentence_id': row.get('sentence_id', '')
        })
    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edges)
    nodes_df.to_csv(outdir/"nodes.csv", index=False)
    edges_df.to_csv(outdir/"edges.csv", index=False)
    # cypher import script (for LOAD CSV)
    cypher = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE;
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:{{row.label}} {{name: row.name}})
RETURN count(*);
"""
    with open(outdir/"import.cypher", "w", encoding="utf8") as fh:
        fh.write(cypher)
    print("Wrote nodes.csv and edges.csv to", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True)
    p.add_argument("--outdir", default="neo4j_import")
    args = p.parse_args()
    main(args)
