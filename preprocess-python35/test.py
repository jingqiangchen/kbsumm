def test15():
    from rouge import Rouge
    rouge=Rouge()
    reference="Poor nations pressurise developed countries into granting trade subsidies."
    summary="Poor nations demand trade subsidies from developed nations."
    scores = rouge.get_scores([reference], [summary])
    for score in scores:
        print(score)
def test21():
    from pyrouge import Rouge155
    from pprint import pprint
    
    ref_texts = {'A': "Poor nations pressurise developed countries into granting trade subsidies."}
    summary_text = "Poor nations demand trade subsidies from developed nations."
    
    
    rouge = Rouge155(rouge_home="/home/test/pyrouge-master/tools/ROUGE-1.5.5",n_words=100)
    score = rouge.score_summary(summary_text, ref_texts)
    pprint(score)

if __name__ == '__main__':
    test15()
    test21()