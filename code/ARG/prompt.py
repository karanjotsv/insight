# COMPLAINT_DETECTION
CD = """
{IMAGE}
TITLE: {TITLE}
REVIEW: {REVIEW}

Here is a product's image, accompanied by the user’s review and the review's title. 

The task is to classify the review as either '0' or '1' based on the provided image and the text. 
A classification of '0' indicates the review is 'non-complaint', meaning the user does not express any dissatisfaction with the product. 
A classification of '1' indicates the review is a 'complaint', where the user is expressing a grievance or issue with the product. 

The output should be either '0' or '1'.

COMPLAINT LABEL: 
"""

# RATIONALE GENERATION
ARG = """
{IMAGE}
TITLE: {TITLE}
REVIEW: {REVIEW}
COMPLAINT LABEL: {LABEL}

Here is a product's image, accompanied by the user’s review, the review's title and the associated label (either '0' or '1'). 

A label of '0' indicates the review is 'non-complaint', meaning the user does not express any dissatisfaction with the product. 
A label of '1' indicates the review is a 'complaint', where the user is expressing a grievance or issue with the product. 

Generate a detailed aspect-based rationale for why the review falls under the given label. 
Ensure the rationale is grounded for the provided label (complaint or non-complaint) and focusses on the key aspects discussed in the review.
The rationale should combine the aspect, description, and impact into a cohesive explanation tied to the assigned label.
The rationale must not include the repetition of the user's review, the review's title and the label.
The rationale should be in a single paragraph.

RATIONALE: 
"""

# COMBINED
COMB = """
{IMAGE}
TITLE: {TITLE}
REVIEW: {REVIEW}

Here is a product's image, accompanied by the user’s review and the review's title. 

The task is to classify the review as either '0' or '1' and then generate a detailed aspect-based rationale for the review's classification based on the provided image and the text. 
A classification of '0' indicates the review is 'non-complaint', meaning the user does not express any dissatisfaction with the product. 
A classification of '1' indicates the review is a 'complaint', where the user is expressing a grievance or issue with the product. 

The classification output should be either '0' or '1'.
Ensure the generated rationale is grounded for the classification label and focusses on the key aspects discussed in the review.
The rationale should combine the aspect, description, and impact into a cohesive explanation tied to the label.

OUTPUT: The review is classified as 
"""

# DESCRIPTION
VER = """
{IMAGE}
TITLE: {TITLE}
REVIEW: {REVIEW}

Examine the given product image with the user’s review and the review's title.
If the review is incomplete or missing details, refer the image to identify additional information about the product, such as issues or positive features.

The task is to enhance the review by making it more descriptive and precise, ensuring it is grounded in both the review's text and the visual elements of the image.
Avoid adding any unrelated details, ensure that the enhanced review remains coherent and accurately portrays the user’s overall experience.

ENHANCED REVIEW:
"""


def set_prompt(CONFIG, IMG_TOKEN, TASK):
    '''
    format prompt for set task(CD or ARG)
    '''
    task_dict = {'CD': CD, 'ARG': ARG, 'BOTH': COMB, 'VER': VER}

    prompt = task_dict[TASK]

    if TASK in ['CD', 'BOTH', 'VER']: 
        return prompt.format(IMAGE=IMG_TOKEN, TITLE=CONFIG['TITLE'], REVIEW=CONFIG['REVIEW'])
    elif TASK == 'ARG':
        return prompt.format(IMAGE=IMG_TOKEN, TITLE=CONFIG['TITLE'], REVIEW=CONFIG['REVIEW'], LABEL=CONFIG['LABEL'])
    else:
        raise Exception('INVALID TASK!')
