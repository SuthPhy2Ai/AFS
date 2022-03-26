# Atomic Feature Set (AFS)  
The supporting materials of cuprate superconducting materials above liquid nitrogen temperature from machine learning  
- ① This software is written as a simple and practical exe program, which does not need other supporting environment, but can be used directly on the Windows system.  
- ② The software is used to extract the characteristics of simple chemical formula, temporarily **does not support** -- **"chemical formula with brackets, hydrate symbols, symbols such as electron valence"** if you have special needs can contact the author, contact information see the console.  
- ③ The limit of features dimension are 1000 (the results follows your feature file), 10000 is the most characters per line, you can not only use the feature file(see elemFeature) of the basic nature of element physics we provide but also customize it to add the features you want.  
- ④ Name the chemical formula file and feature file as data.csv and Fillfeature.csv respectively. Make sure your files are encoded in **UTF-8**, otherwise the first line may be garbled.  
- ⑤ Ensure Fillfeature.csv (feature file) and data.csv (require only the column of chemical formula and no other redundant characters,see example) in the same folder as AFS.exe.   
- ⑥ Double click AFS.exe , and select the mode for extracting features **input: ICQMSicqms**, so get the out.csv file.  


| Feature | Specific Meaning |   
| :----:| :----: |  
| Number | The atomic number of the element |  
| MendeleevNumber | The mendeleev number of the element |  
| AtomicWeight | Mass of the atom |  
