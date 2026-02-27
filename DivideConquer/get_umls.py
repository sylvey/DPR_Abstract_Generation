class GetUMLS:
    
    def __init__(self, nlp, linker):
        self.nlp = nlp
        self.linker = linker

    def get_umls_with_original_text(self, text, top_n=5):
        doc = self.nlp(text)
        term_map = {} # Key: Canonical Name, Value: {original_texts: set(), rank: float}

        for phrase in doc._.phrases:
            for ent in phrase.chunks:
                if ent._.kb_ents:
                    cui = ent._.kb_ents[0][0]
                    canon_name = self.linker.kb.cui_to_entity[cui].canonical_name
                    
                    if canon_name not in term_map:
                        term_map[canon_name] = {"original": set(), "rank": phrase.rank}
                    
                    term_map[canon_name]["original"].add(ent.text)
                    term_map[canon_name]["rank"] = max(term_map[canon_name]["rank"], phrase.rank)

        sorted_items = sorted(term_map.items(), key=lambda x: x[1]['rank'], reverse=True)[:top_n]
        
        formatted_terms = []
        for name, info in sorted_items:
            orig = list(info['original'])[0] 
            if name.lower() == orig.lower():
                formatted_terms.append(name)
            else:
                formatted_terms.append(f"{name} (referenced as '{orig}')")
        
        return formatted_terms

    def get_umls_with_original_text_mesh(self, text, top_n=5, weight=1.5): # 建議設為 10
        doc = self.nlp(text)
        term_map = {} 

        for phrase in doc._.phrases:
            for ent in phrase.chunks:
                if ent._.kb_ents:
                    cui, score = ent._.kb_ents[0]
                    entity_info = self.linker.kb.cui_to_entity[cui]
                    canon_name = entity_info.canonical_name
                    
                    mesh_like_tuis = {
                        "T047", "T191", "T059", "T060", "T121", "T033", 
                        "T061", "T023", "T048", "T046" 
                    }
                    is_mesh_type = any(tui in mesh_like_tuis for tui in entity_info.types)

                    rank_score = phrase.rank
                    if is_mesh_type:
                        rank_score *= weight
                    
                    if canon_name not in term_map:
                        term_map[canon_name] = {"original": set(), "rank": rank_score}
                    
                    term_map[canon_name]["original"].add(ent.text)
                    term_map[canon_name]["rank"] = max(term_map[canon_name]["rank"], rank_score)

        sorted_items = sorted(term_map.items(), key=lambda x: x[1]['rank'], reverse=True)[:top_n]
        
        formatted_terms = []
        for name, info in sorted_items:
            orig = min(list(info['original']), key=len)
            if name.lower() == orig.lower():
                formatted_terms.append(name)
            else:
                formatted_terms.append(f"{name} (referenced as '{orig}')")
        
        return formatted_terms
    
    def get_umls_with_textrank(self, text, top_n=5):
        doc = self.nlp(text)
        
        term_weights = {}

        for phrase in doc._.phrases:
            for ent in phrase.chunks:
                if ent._.kb_ents:
                    cui = ent._.kb_ents[0][0] 
                    canonical_name = self.linker.kb.cui_to_entity[cui].canonical_name
                    
                    if canonical_name not in term_weights or phrase.rank > term_weights[canonical_name]:
                        term_weights[canonical_name] = phrase.rank

        sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
        return [term for term, rank in sorted_terms[:top_n]]