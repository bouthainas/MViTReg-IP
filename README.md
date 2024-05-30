# MViTReg-IP Lung Analysis Architecture
![ ](https://github.com/bouthainas/MViTReg-IP/blob/main/MViTReg-IP.png)
## What is MViTReg-IP?
MVitReg-IP is a multi-task deep learning algorithm that can quickly and effectively determine the severity of lung infection in patients with COVID-19 or similar lung diseases. This model is able to predict scores associated with different disease courses and the severity of lung infections. It is based on a dual transformer encoder, followed by a feature fusion module feeding two MLP regression heads to quantify CXR images with two scores simultaneously. One score is based on the spread of infection in the lungs, while the other measures how opaque the infected sites are. 
http://dx.doi.org/10.1109/TETCI.2024.3359082

# Team
## Core Contributors
* Biomedical Engineer, PhD Student, Bouthaina Slika, University of the Basque Country, Spain, bslika001@ikasle.ehu.eus
* Prof. Dr. Fadi Dornaika, IEEE member, Dept. of Artificial Intelligence, University of the Basque Country & IKERBAQUE foundation, Spain fadi.dornaika@ehu.es
* Prof. Dr. Karim Hammoudi, IEEE member, Group Imagery, Dept. of Computer Science, IRIMAS, Université de Haute-Alsace, France, karim.hammoudi@uha.fr
* Medical Doctor, PhD, Hamid Merdji, French National Institute of Health and Medical Research (INSERM), Regenerative Nanomedicine (RNM), Biomedicine Research Center (CRBS), Federation of Translational Medicine, and Dept. of Intensive Medicine-Resuscitation, Hospital of Strasbourg, France, merdgi.hamid@gmail.com
* Dr. Vinh Truong Hoang, Dept. of Computer Science, HCMC Open University, Ho Chi Minh City, Vietnam, vinth@ou.edu.vn
* Iyed Dhahri, Final Year Engineering Student, ENSI, Tunisia

## Collaborators
* Prof. Dr. Halim Benhabiles, Group BIO-MEMS, Dept. of Artificial Intelligence, JUNIA, CNRS, IEMN, University of Lille, halim.benhabiles@yncrea.fr
* Prof. Dr. Mahmoud Melkemi, Group Imagery, Dept. of Computer Science, IRIMAS, Université de Haute-Alsace, mahmoud.melkemi@uha.fr
* Prof. Dr. Adnance Cabani, Dept. of Computer Science, ESIGELEC/IRSEEM, Normandy University, adnane.cabani@esigelec.fr

# Reference
> Karim Hammoudi, Halim Benhabiles, Mahmoud Melkemi, Fadi Dornaika, Ignacio Arganda-Carreras, Dominique Collard, and Arnaud Scherpereel. Deep learning on chest x-ray images to detect and evaluate pneumonia cases at the era of covid-19. Journal of medical systems, 45(7):1–10, 2021. https://doi.org/10.1007/s10916-021-01745-4
```
@Article{Hammoudi2021,
author={Hammoudi, Karim
and Benhabiles, Halim
and Melkemi, Mahmoud
and Dornaika, Fadi
and Arganda-Carreras, Ignacio
and Collard, Dominique
and Scherpereel, Arnaud},
title={Deep Learning on Chest X-ray Images to Detect and Evaluate Pneumonia Cases at the Era of COVID-19},
journal={Journal of Medical Systems},
year={2021},
month={June},
day={08},
issn={1573-689X},
doi={10.1007/s10916-021-01745-4},
url={https://doi.org/10.1007/s10916-021-01745-4}
}
```
> Bouthaina Slika, Fadi Dornaika, Karim Hammoudi, and Vinh Truong Hoang. Automatic quantification of lung infection severity in chest x-ray images. In IEEE Statistical Signal Processing (SSP) Workshop, pages 418–422. IEEE, 2023.
```
@inproceedings{slika2023ssp,
title={Automatic Quantification of Lung Infection Severity in Chest X-ray Images},
author={Slika, Bouthaina
and Dornaika, Fadi
and Hammoudi, Karim
and Hoang, Vinh Truong},
booktitle={IEEE Statistical Signal Processing (SSP) Workshop},
pages={418--422},
year={2023},
organization={IEEE}
}
```
![ ](https://github.com/bouthainas/ViTReg-IP/blob/main/Contributors.png)
