import PyPDF2
from tqdm import tqdm
import re

dashed_line = "--------------------------------------------------------------------------------"
dashed_line_errors = ["-----------------------------------------------------------------------------If", "----------------------------------------------------------------------------How"]
cols = "                           Freq.   Numeric  Label                          "


def scrape_guide(filepath="dfvsg_Guide_May2018_VOTER_Survey.pdf"):
    pdfFileObj = open(filepath, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    full_text = ""
    for page_num in tqdm(range(pdfReader.numPages), desc="Extracting Text from PDF"):
        pageObj = pdfReader.getPage(page_num)
        page_text = pageObj.extractText()
        page_parts = page_text.split('"')
        for part in page_parts:
            if dashed_line not in part:
                page_parts.remove(part)
        for part in page_parts:
            full_text += part
    for error in dashed_line_errors:
        full_text = full_text.replace(error, dashed_line)
    text_splitted = full_text.split((dashed_line))
    return text_splitted

def variable_labeler(raw_variable):
    slimmed_variable = ' '.join(raw_variable.split()) # Source: https://stackoverflow.com/a/2077944
    variable_pair = slimmed_variable.split(" ", maxsplit=1)
    if len(variable_pair) == 1:     #generalized debug of `regzip_baseline`, which is unlabeled
        variable_pair.append(variable_pair[0])
    if variable_pair[1] == '(unlabeled)':
        variable_pair[1] = variable_pair[0]
    return variable_pair[0], variable_pair[1]

def response_labeler(raw_q_and_a, code):
    if cols in raw_q_and_a:
        q_and_a = raw_q_and_a.split(cols)
    else:
        return raw_q_and_a, ' '

    question = q_and_a[0].strip()

    q_and_a[1] = ' '.join(q_and_a[1].split())
    q_and_a[1] = q_and_a[1].replace(",","")

    freqs_and_nos = re.findall('\d+', q_and_a[1])
    freqs_and_nos.append(-1)
    freqs_and_nos = list(zip(freqs_and_nos[0::2],freqs_and_nos[1::2]))

    #label_values = []
    labels = (re.split('\d+', q_and_a[1]))   #Not perfect ... numbers in response, e.g. 2008, will break?
    #print(label_values)
    labels = list(filter(lambda a: a != ' ', labels))
    labels = list(filter(lambda a: a != '', labels))
    labels = [a.strip() for a in labels]
    #q_and_a[1] = label_values

    if len(freqs_and_nos) < len(labels):
    #if len(freqs_and_nos) > len(labels):  <-- should be this
        items = len(freqs_and_nos)
    else:
        items = len(labels)

    key_items = []
    for item in range(items):
        key_item = {'freq':freqs_and_nos[item][0], 'numeric':int(freqs_and_nos[item][1]), 'label' : labels[item]}
        key_items.append(key_item)
    #if (len(freqs_and_nos) == len(labels)) == False:
        #print(code, len(freqs_and_nos), len(labels))

    return question, key_items

def compile_dictionary(filepath="dfvsg_Guide_May2018_VOTER_Survey.pdf"):
    text_splitted = scrape_guide(filepath)
    guides = {}
    raw_guides = list(zip(text_splitted[1::2],text_splitted[2::2]))
    for raw_guide in raw_guides:
        #print(raw_guide)
        code, label = variable_labeler(raw_guide[0])
        question, answer = response_labeler(raw_guide[1], code)
        guide = {code: {'variable_label' : label,
                      'question' : question,
                      'responses' :answer
                      }}
        guides.update(guide)
    print("👍 Survey guide PDF file scraped and compiled into Python-readable dictionary.")
    return guides


###########################
# Feature Labeler         #
###########################

important_features = ["case_identifier",
    "caseid",
    "weight_panel",
    "weight_latino",
    "weight_18_24",
    "weight_overall",
    "cassfullcd",
    "starttime",
    "endtime"]

important_features_dict = {
    'case_identifier' : 'Case Identifier',
    'caseid' : 'Case ID',
    'weight_panel' : 'Statistical Weight (Panel)',
    'weight_latino' : 'Statistical Weight (Latino)',
    'weight_18_24' : 'Statistical Weight (18-24 yrs.)',
    'weight_overall' : 'Statistical Weight (Overall)',
    'cassfullcd' : 'cassfullcd', #Not sure what this is?
    'starttime' : 'Start Time',
    'endtime' : 'End Time'
}

def concise_variables(variables):
    concise_variables = []
    for variable in variables:
        expanded_variable = variable.split(" - ")
        concise_variables.append(expanded_variable[-1])
    return concise_variables


def feature_labeler(voters_data, guide_dict, make_concise = True):
    variables = voters_data.columns
    variable_labels = {i:guide_dict[i]['variable_label'] for i in list(guide_dict)}
    variable_labels.update(important_features_dict)
    variables = variables.map(variable_labels)
    if make_concise is True:
        variables = concise_variables(variables)
    print('🏷️ {} variables labeled.\nFor example: "{}" → "{}"'.format(len(variables), list(voters_data)[0], list(variables)[0]))
    return list(variables)
