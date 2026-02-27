



class ParagraphPrompts:
    @staticmethod
    def prompt1(sec_norm, sentence_length):
        return f"""You are a biomedical {sec_norm if sec_norm != "none" else "general"} paragraph summarization assistant. The summary should be {sentence_length[sec_norm]} sentences long."""
    def prompt2(sec_norm, sentence_length):
        return f"""You are a biomedical {sec_norm if sec_norm != "none" else "general"} paragraph summarization assistant. The summary should be {sentence_length[sec_norm]} sentences long.
    CRITICAL INSTRUCTION:
    Respond ONLY with a valid JSON object. 
    Do NOT use Markdown code blocks (like ```json). 
    Do NOT provide any conversational text.
    
    Format:
    {{
        "summary": "your summary text here",
        "reasoning": "your reasoning here"
    }}
    """
    def prompt3(sec_norm, sec_text, compression_ratio=26.2):
        """
        Sectional Prompt with compression rate
        """
        word_count = len(sec_text.split())
        target_words = max(15, int(word_count / compression_ratio))
        
        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        section_guidelines = {
            "background": "Focus on research gap and motivation.",
            "objective": "State the primary aim or hypothesis.",
            "methods": "Detail study design and procedures.",
            "results": "Prioritize key findings and data.",
            "conclusions": "Summarize implications and take-home messages.",
            "none": "Focus on the main point of the paragraph.",
            "intro": "Introduce the main topic and context.",
            "main idea": "Focus on the central concept or hypothesis.",
            "result and conclusion": "Focus on key findings and their implications."   
        }
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this {sec_norm if sec_norm != 'none' else 'general'} section in about {target_words} words.\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    def prompt4(sec_norm, sec_text, compression_ratio=26.2):
        """
        Simple split prompt with compression rate
        """
        word_count = len(sec_text.split())
        target_words = max(15, int(word_count / compression_ratio))
        
        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        
        specific_guide = "Summarize the main topic."

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this section in about {target_words} words.\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    def prompt5(sec_norm, sec_text):
        """
        Sectional Prompt with auto length
        """
        
        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        section_guidelines = {
            "background": "Focus on research gap and motivation.",
            "objective": "State the primary aim or hypothesis.",
            "methods": "Detail study design and procedures.",
            "results": "Prioritize key findings and data.",
            "conclusions": "Summarize implications and take-home messages.",
            "none": "Focus on the main point of the paragraph.",
            "intro": "Introduce the main topic and context.",
            "main idea": "Focus on the central concept or hypothesis.",
            "result and conclusion": "Focus on key findings and their implications."   
        }
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this {sec_norm if sec_norm != 'none' else 'general'} section\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    def prompt6(sec_norm, sec_text):
        """
        Simple split prompt with auto length
        """
        
        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        
        specific_guide = "Summarize the main topic."

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this section.\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    def prompt7(sec_norm, sec_text, get_umls_terms_textRank, top_umls_terms):
        """
        Sectional prompt with auto length and UMLS guidedance.
        """


        top_entities = get_umls_terms_textRank(sec_text, top_umls_terms)

        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        section_guidelines = {
            "background": "Focus on research gap and motivation.",
            "objective": "State the primary aim or hypothesis.",
            "methods": "Detail study design and procedures.",
            "results": "Prioritize key findings and data.",
            "conclusions": "Summarize implications and take-home messages.",
            "none": "Focus on the main point of the paragraph.",
            "intro": "Introduce the main topic and context.",
            "main idea": "Focus on the central concept or hypothesis.",
            "result and conclusion": "Focus on key findings and their implications."   
        }
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this {sec_norm if sec_norm != 'none' else 'general'} section\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Relevant UMLS Concepts: {', '.join(top_entities)}\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg

    def prompt8(sec_norm, sec_text):
        """
        Sectional Prompt with auto length (using comprehensive direction instead of summarization focus)
        """
        
        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        section_guidelines = {
            "background": "Focus on research gap and motivation.",
            "objective": "State the primary aim or hypothesis.",
            "methods": "Detail study design and procedures.",
            "results": "Prioritize key findings and data.",
            "conclusions": "Summarize implications and take-home messages.",
            "none": "Focus on the main point of the paragraph.",
            "intro": "Introduce the main topic and context.",
            "main idea": "Focus on the central concept or hypothesis.",
            "result and conclusion": "Focus on key findings and their implications."   
        }
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical synthesis assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Synthesize the critical information from this {sec_norm if sec_norm != 'none' else 'general'} section into a dense, informative paragraph.\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    def prompt9(sec_norm, sec_text, get_umls_terms_textRank, top_umls_terms):
        """
        Sectional prompt with auto length and UMLS guidedance. (Refined with https://support.jmir.org/hc/en-us/articles/37982552280987-Submitting-Your-Manuscript-to-JMIR-Publications-A-Guide-for-Authors)
        """


        top_entities = get_umls_terms_textRank(sec_text, top_umls_terms)

        rules = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )

        section_guidelines = {
            "background": "Briefly describe the context and significance of the research.",
            "objective": "State the specific aim(s) of the study in a complete sentence.",
            "methods": "Outline the research design, study sample, data collection, and analysis procedures.",
            "results": "Present key findings, including relevant statistics (sample sizes, response rates, P values, confidence intervals). Be specific.",
            "conclusions": "Summarize the main findings and their implications.",
            "none": "Synthesize the core biomedical information, focusing on the primary argument, concept, or supplementary context presented.",
            "intro": "Describe the research context and significance, and clearly state the specific aims or hypotheses.",
            "main idea": "Outline the research design and procedures, while capturing any supplementary methodological context or core concepts.",
            "result and conclusion": "Present key findings with relevant statistics, and summarize their broader implications and take-home messages."   
        }
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical synthesis assistant. "
            f"{rules}"
            "Define your output strictly as:\n"
            "- 'summary': The synthesized biomedical text.\n"
            "- 'reasoning': A brief explanation of how your summary fulfills the given instructions.\n"
            'Format: {"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Synthesize the critical information from this {sec_norm if sec_norm != 'none' else 'general'} section provided in the 'Paragraph text'\n"
            f"{specific_guide}\n\n"
            f"Ensure the core meanings of these key biomedical entities are preserved or synthesized accurately: {', '.join(top_entities)}\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg

class AbstractPrompts:
    @staticmethod
    def prompt1():
        return f"""You are a biomedical abstract refinement assistant. Refine the abstract based on the abstract draft."""
    def prompt2():
        return f"""You are a biomedical abstract refinement assistant. Refine the abstract based on the abstract draft.
    CRITICAL INSTRUCTION:
    Respond ONLY with a valid JSON object. 
    Do NOT use Markdown code blocks (like ```json). 
    Do NOT provide any conversational text.
    
    Format:
    {{
        "abstract": "your abstract text here",
        "reasoning": "your reasoning here"
    }}
    """