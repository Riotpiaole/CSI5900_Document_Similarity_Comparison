# Report

## Definition of Problem

- Classified document based on different type of documents 
  
  1. images jpg png format
  
  2. PDF portable Document format.

## Approaches

### Image Format

- One document as an image has various format. For classified one image is belong to a specific class or types of author or organization. Distinctively, this task is complete by human require some interesting features from one document such as context of the document and layout of the document.

  - Thus this raised the first questions

  - What can represents a document for other to distinguish.

    - Layouts: A document consists various types of region: Text, Title, SubTitle, Figure and table. These are some interesting features can used to characterized a document. This problem can be described as performing object detection problem to an image which yield to predict maximum accuracy for category of layouts.

      - Some documents where created by specific group of organization will used a specific types of layout. For example, various document form will follow similarly despite different version changes. 

- In specifically, the level of document also diverge as the development of businesses and society. Zejiang Shen suggested document layouts can be distinguish between modern scientific documents, scientific report, news paper, table based business document such as case studies, anneal and quarter report[paper](https://arxiv.org/pdf/2103.15348.pdf)

### Text Format:

- As critical features that is extracted from different types of layout of the document is the text region. These region contains various NLP feature that comprised the more concrete feature that distinguish one document from another.