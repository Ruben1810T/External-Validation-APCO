v = Vital_data
h = Hemosphere data 
^^ Zijn verschillende monitoren

De code is gebasseerd op het gebruik van data uit het MST. Zo worden pathways gebruikt die specifiek zijn voor de apparatuur die gebruikt is. 
Namelijk: 
- MARTINI
- Intellivue Wat de patientspecifieke bewaking is van philips.

Ook de manier hoe de tijd verwerkt wordt is specifiek.

De code is aangepast op de data die wordt verwerkt. Zo bestaat de v data uit minutes en Intellivue kolommen. Dit is zo aangegeven in de code

De h data is langer dan de v data. 

De plots die gemaakt worden kan je aan en uit zetten in de vars.env

De lowess smoothing filter genaamd: "lowess_smoothing" is gekopieerd van de code van, van Mierlo. De tweede defenitie: "lowess_sv" is een toepassing op dit filter

In de code worden filters van van Mierlo gebruikt. Echter passen wij extra filter stappen toe. Deze kunnen aan of uit gezet worden in de instellingen.