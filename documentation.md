# Report

## Definition of Problem

- Classified document based on different type of documents 
  
  1. images jpg png format
  
  2. PDF portable Document format.

## Related Work

1. [Combining Visual and Textual Features for Semantic Segmentation of Historical Newspaper](https://arxiv.org/pdf/2002.06144.pdf)

   - Segmenting the text into word vector representation and 3D positions and embedding representation

     - Segmented optical character recognized text and feed to into embedding based nn

     - Image based feature feeding into Convolutional NN

2. [VisualWordGrid: Information Extraction From Scanned Documents Using A Multimodal Approach](https://arxiv.org/pdf/2010.02358v5.pdf)

  1. Seperating two models for two purposes

     1. extracted text features
     2. Embedded model for NLP feature analysis.

  2. OCR or PDF parser

     1. This can concretely extracted text

### Image Format


#### Layout Similarity VIA

- One document as an image has various format. For classified one image is belong to a specific class or types of author or organization. Distinctively, this task is complete by human require some interesting features from one document such as context of the document and layout of the document.

  - Thus this raised the first questions

  - What can represents a document for other to distinguish.

    - Layouts: A document consists various types of region: Text, Title, SubTitle, Figure and table. These are some interesting features can used to characterized a document. This problem can be described as performing object detection problem to an image which yield to predict maximum accuracy for category of layouts.

      - Some documents where created by specific group of organization will used a specific types of layout. For example, various document form will follow similarly despite different version changes.

- In specifically, the level of document also diverge as the development of businesses and society. Zejiang Shen suggested document layouts can be distinguish between modern scientific documents, scientific report, news paper, table based business document such as case studies, anneal and quarter report[paper](https://arxiv.org/pdf/2103.15348.pdf)

- The layout extracted from document can later convert to reducible syntactical tree and compare based on similarity. This can carefully examinee the visual similarity of each document.

### Linguistic Feature Representation:

- Another critical features that is extracted from a document is the text region. These region contains various NLP feature that comprised the more concrete feature that distinguish one document from another.

- Human observation comprehend a document by understanding the characters, words, paragraphs and layout component. As previous section had discussed the method of layout analysis, This section is aimed to discuss the method to derive nlp features from text within layout. This can serve as additional feature for the overall architecture.

 asdasd