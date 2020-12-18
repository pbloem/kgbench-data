import hdt

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os, tqdm, gzip
from collections import Counter
import pandas as pd
from unidecode import unidecode

import rdflib as rdf

import kgbench as kg
from kgbench import tic, toc

whitelist = {

    # DBLP relations
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://xmlns.com/foaf/0.1/maker',
    'http://purl.org/dc/elements/1.1/subject',
    'http://swrc.ontoware.org/ontology#editor',
    'http://schema.org/description',
    'http://www.w3.org/2004/02/skos/core#altLabel',
    'http://swrc.ontoware.org/ontology#month',
    'http://purl.org/dc/terms/issued',
    'http://www.w3.org/2002/07/owl#sameAs',
    'http://xmlns.com/foaf/0.1/name',
    'http://swrc.ontoware.org/ontology#series',
    'http://swrc.ontoware.org/ontology#journal',
    'http://purl.org/dc/elements/1.1/publisher',
    'http://purl.org/dc/elements/1.1/type',
    'http://www.w3.org/2004/02/skos/core#prefLabel',
    'http://xmlns.com/foaf/0.1/homepage',
    'http://purl.org/dc/terms/references',
    'http://purl.org/dc/terms/partOf',
    'http://swrc.ontoware.org/ontology#number',
    'http://purl.org/dc/elements/1.1/title',
    'http://swrc.ontoware.org/ontology#volume',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://schema.org/name',

    'http://www.w3.org/2000/01/rdf-schema#seeAlso', # contains DOI

    # Wikidata relations with person as subject
    # 'http://www.wikidata.org/prop/direct/P5029', # "Researchmap ID"@en "іdentifier for a researcher on researchmap.jp"@en
    # 'http://www.wikidata.org/prop/direct/P1005', # "Portuguese National Library ID"@en "identifier for the Portuguese National Library"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4419', # "Videolectures ID"@en "identifier for person, meeting or presentation/talk (video) on the Videolectures website"@en
    # 'http://www.wikidata.org/prop/direct/P2953', # "Estonian Research Portal person ID"@en "identifier for a person, in the Estonian Research Portal"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P1617', # "BBC Things ID"@en-gb "identifier in the BBC Things database"@en-gb
    'http://www.wikidata.org/prop/direct/P1477', # "birth name"@en-gb "full name of a person at birth, if different from their current, generally used name (samples: John Peter Doe for Joe Doe, Ann Smith for Ann Miller)"@en-ca
    # 'http://www.wikidata.org/prop/direct/P5361', # "BNB person ID"@en-gb "identifier of a person in the British National Bibliography (bnb.data.bl.uk)"@en-gb
    'http://www.wikidata.org/prop/direct/P856', # "official website"@en-gb "URL of the official homepage of the item"@en-ca
    # 'http://www.wikidata.org/prop/direct/P4440', # "Biblioteca Nacional de México ID"@en "authority control identifier used at the Biblioteca Nacional de México"@en
    # 'http://www.wikidata.org/prop/direct/P3218', # "Auñamendi ID"@en "identifier of an item in Auñamendi Encyclopaedia"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P950', # "Biblioteca Nacional de España ID"@en "identifier from the authority file of the Biblioteca Nacional de España. Format for persons: "XX" followed by 4 to 7 digits"@en
    # 'http://www.wikidata.org/prop/direct/P1344', # "participant of"@en-gb "event a person or an organization was/is a participant in, inverse of P710 or P1923"@en
    # 'http://www.wikidata.org/prop/direct/P6585', # "Directorio Legislativo ID"@en ""Directorio Legislativo" is a well know NGO in Argentina who has been publishing for the last ten years a book (and now online) with information about every deputy in the National Congress."@en
    'http://www.wikidata.org/prop/direct/P140', # "religion"@en-gb "religion of a person, organization or religious building, or associated with this subject"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4527', # "UK Parliament thesaurus ID"@en-gb "Thesaurus of subject headings maintained by the libraries of the House of Commons and House of Lords in the UK. Covers around 110,000 topics (concepts, people, places, organisations, legislation, etc), with a structured taxonomy."@en-gb
    # 'http://www.wikidata.org/prop/direct/P866', # "Perlentaucher ID"@en "identifier in Perlentaucher (Q2071388)"@en
    # 'http://www.wikidata.org/prop/direct/P5068', # "Flanders Arts Institute person ID"@en "identifier of a person in the Flanders Arts Institute database for performing arts"@en
    # 'http://www.wikidata.org/prop/direct/P2006', # "ZooBank author ID"@en-gb "identifier for an author at ZooBank"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2734', # "Unz Review author ID"@en "author identifier at The Unz Review (unz.org), a content-archiving website"@en
    # 'http://www.wikidata.org/prop/direct/P1455', # "list of works"@en "link to the article with the works of a person"@en
    # 'http://www.wikidata.org/prop/direct/P4663', # "DACS ID"@en "code to identify 50,000 artists as members of the British collective rights management organisation DACS and sister organisations worldwide"@en
    'http://www.wikidata.org/prop/direct/P463',
    # "member of"@en-gb "organization, musical group, or club to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a position such as a member of parliament (use P39 for that)."@en
    # 'http://www.wikidata.org/prop/direct/P3348', # "National Library of Greece ID"@en "authority ID from the National Library of Greece Authority Records"@en
    # 'http://www.wikidata.org/prop/direct/P1960', # "Google Scholar author ID"@en "identifier of a person, in the Google Scholar academic search service"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P2581', # "BabelNet ID"@en-gb "ID in BabelNet encyclopedic dictionary"@en-gb
    # 'http://www.wikidata.org/prop/direct/P549', # "Mathematics Genealogy Project identifier"@en-gb "identifier for mathematicians and computer scientists at the Mathematics Genealogy Project"@en-gb
    # 'http://www.wikidata.org/prop/direct/P949', # "National Library of Israel ID"@en "identifier for authority control used at the National Library of Israel"@en
    # 'http://www.wikidata.org/prop/direct/P269', # "SUDOC identifier"@en-gb "identifier for authority control in the French collaborative library catalog (see also P1025). Format: 8 digits followed by a digit or "X""@en
    # 'http://www.wikidata.org/prop/direct-normalized/P2732', # "Persée author ID"@en "identifier for an author, in Persée"@en
    # 'http://www.wikidata.org/prop/direct/P5337', # "Google News topics ID"@en-gb "identifier for a subject in the news on Google News"@en
    'http://www.wikidata.org/prop/direct/P1412',
    # "languages spoken, written or signed"@en-gb "language(s) that a person speaks, writes or signs, including the native language(s)"@en
    # 'http://www.wikidata.org/prop/direct/P3430', # "SNAC Ark ID"@en "identifier for items in the Social Networks and Archival Context system"@en
    # 'http://www.wikidata.org/prop/direct/P1186', # "MEP directory ID"@en "identifier for a past or present MEP in a directory of all members of the European Parliament"@en
    # 'http://www.wikidata.org/prop/direct/P4890', # "EPHE ID"@en "identifier of a researcher on the online prosopographical dictionary of the EPHE"@en
    # 'http://www.wikidata.org/prop/direct/P3124', # "Polish scientist ID"@en "identifier for a scientist, in the Polish Government's Information Processing Centre database"@en
    # 'http://www.wikidata.org/prop/direct/P3608', # "EU VAT number"@en "EU VAT number"@en
    # 'http://www.wikidata.org/prop/direct/P1814', # "name in kana"@en "the reading of a Japanese name in kana"@en
    # 'http://www.wikidata.org/prop/direct/P2397', # "YouTube channel ID"@en-gb "ID of the YouTube channel of a person or organisation (not to be confused with the name of the channel)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P6815', # "University of Amsterdam Album Academicum ID"@en "identifier of professors and doctors of the University of Amsterdam and its predecessor, the Athenaeum Illustre, from 1632 up to now"@en
    # 'http://www.wikidata.org/prop/direct/P1441', # "present in work"@en-gb "work in which this fictional entity (Q14897293) or historical person is present (use P2860 for works citing other works, P361/P1433 for works being part of / published in other works, P1343 for entities described in non-fictional accounts)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3846', # "DBC author ID"@en "identifier for authors set by the Danish Bibliographic Centre"@en
    # 'http://www.wikidata.org/prop/direct/P2171', # "TheyWorkForYou ID"@en-gb "identifier in the 'TheyWorkForYou' database of British MPs"@en-gb
    # 'http://www.wikidata.org/prop/direct/P7748', # "NIPS Proceedings author ID"@en "identifier for an author publishing or editor at the NIPS/NeurIPS conference"@en
    # 'http://www.wikidata.org/prop/direct/P1045', # "Sycomore ID"@en "identifer in the Sycomore database of French MPs, National Assembly (France)"@en
    'http://www.wikidata.org/prop/direct/P802',  # "student"@en-gb "notable student(s) of an individual"@en
    # 'http://www.wikidata.org/prop/direct/P5320', # "IUF member ID"@en-gb None
    # 'http://www.wikidata.org/prop/direct/P39', # "position held"@en "subject currently or formerly holds the object position or public office"@en
    # 'http://www.wikidata.org/prop/direct/P244', # "Library of Congress authority ID"@en "only for authority control"@en-ca
    'http://www.wikidata.org/prop/direct/P1559',
    # "name in native language"@en-gb "name of a person in their native language. Could be displayed in addition to the label, if language has a different script"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3267', # "Flickr user ID"@en "identifier for a person or organisation, with an account at Flickr"@en
    # 'http://www.wikidata.org/prop/direct/P443', # "pronunciation audio"@en-gb "audio file with pronunciation"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P349', # "National Diet Library Auth ID"@en "identifier for authority control per the National Diet Library of Japan"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P1006', # "NTA identifier (Netherlands)"@en-gb "identifier for persons (not:works) from the Dutch National Thesaurus for Author names (which also contains non-authors)"@en
    # 'http://www.wikidata.org/prop/direct/P1996', # "parliament.uk biography pages"@en "link to an MP or Peer's biography on parliament.uk"@en
    'http://www.wikidata.org/prop/direct/P6553',
    # "preferred pronoun"@en-gb "personal pronoun(s) this person uses"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3509', # "Dagens Nyheter topic ID"@en "identifier for a topic, used by the Swedish daily newspaper Dagens Nyheter"@en
    # 'http://www.wikidata.org/prop/direct/P1942', # "McCune-Reischauer romanization"@en "romanization system for Korean"@en
    # 'http://www.wikidata.org/prop/direct/P345', # "IMDb ID"@en-gb "identifier for the IMDb [with prefix 'tt', 'nm', 'co', 'ev', 'ch' or 'ni']"@en
    # 'http://www.wikidata.org/prop/direct/P7293', # "NLP ID (PLWABN unique record)"@en "National Library of Poland record no. identifier. Format: "981", followed by 8 digits, then ending with "05606""@en
    # 'http://www.wikidata.org/prop/direct/P6556', # "SICRIS researcher ID"@en "ID of a registered researcher in Slovenia from the Slovenian Current Research Information System (SICRIS)"@en
    'http://www.wikidata.org/prop/direct/P26',
    # "spouse"@en-gb "the subject has the object as their spouse (husband, wife, partner, etc.). Use "unmarried partner" (P451) for non-married companions"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1317', # "floruit"@en-gb "when the person was known to be active or alive, when birth or death not documented (contrast with P585 and P746). Can have multiple values if the subject has produced works in multiple years"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P434', # "MusicBrainz artist ID"@en-gb "identifier for an artist in the MusicBrainz open music encyclopædia"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5664', # "Savoirs ENS ID"@en "identifier for a lecturer on Savoirs ENS"@en
    # 'http://www.wikidata.org/prop/direct/P2930', # "INSPIRE-HEP author ID"@en-gb "identifier for authors in INSPIRE-HEP, a major database for high energy physics"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1950', # "second family name in Spanish name"@en "second (generally maternal) family name in Spanish names (do not use for other double barrelled names)"@en
    # 'http://www.wikidata.org/prop/direct/P373', # "Commons category"@en-gb "name of the Wikimedia Commons category containing files related to this item (without the prefix "Category:")"@en
    # 'http://www.wikidata.org/prop/direct/P1741', # "GTAA ID"@en "identifier for GTAA, a thesaurus used in audiovisual archives (NISV, EYE)"@en
    # 'http://www.wikidata.org/prop/direct/P6023', # "ResearchGate contributor ID"@en-gb "identifier for a non-registered author on ResearchGate"@en-gb
    'http://www.wikidata.org/prop/direct/P2031',
    # "work period (start)"@en-gb "start of period during which a person or group flourished (fl. = "floruit") in their professional activity"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1871', # "CERL ID"@en-gb "identifier in the Consortium of European Research Libraries thesaurus"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P227', # "GND identifier"@en-gb "international authority file of names, subjects, and organisations (please don't use type n = name, disambiguation)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4342', # "Store norske leksikon ID"@en "identifier of an article in the online encyclopedia snl.no"@en
    # 'http://www.wikidata.org/prop/direct/P6100', # "YÖK Academic Profile ID"@en "identifer for an academic, in the Turkish 'YÖK' database"@en
    # 'http://www.wikidata.org/prop/direct/P5019', # "Brockhaus Enzyklopädie online ID"@en "identifier for an article in the online version of Brockhaus Enzyklopädie"@en
    # 'http://www.wikidata.org/prop/direct/P4862', # "Amazon author ID"@en-gb "author identifier on Amazon.com"@en
    'http://www.wikidata.org/prop/direct/P664',  # "organizer"@en "person or institution organizing an event"@en-gb # 'http://www.wikidata.org/prop/direct-normalized/P268', # "BnF ID"@en-gb "identifier for the subject issued by BNF (Bibliothèque nationale de France). Format: 8 digits followed by a check-digit or letter, do not include the initial 'cb'."@en-gb
    # 'http://www.wikidata.org/prop/direct/P2016', # "Catalogus Professorum Academiae Groninganae id"@en "identifier for a professor, in the Catalogus Professorum Academiae Groninganae"@en
    # 'http://www.wikidata.org/prop/direct/P6640', # "JRC Names id"@en "ID in the JRC Names gazetteer, which provides spelling variants and EMM news about the entity (220k news items per day). Alias: JRCN"@en
    'http://www.wikidata.org/prop/direct/P101',
    # "field of work"@en-gb "specialization of a person or organization; see P106 for the occupation"@en-ca
    # 'http://www.wikidata.org/prop/direct/P3219', # "Encyclopædia Universalis ID"@en "identifer for an article in the online version of Encyclopædia Universalis"@en
    # 'http://www.wikidata.org/prop/direct/P1670', # "Canadiana Authorities ID"@en "obsolete identifier for authority control per the Library and Archives Canada. Format: 4 digits + 1 letter + 4 digits"@en
    #'http://www.wikidata.org/prop/direct/P5008',  # "on focus list of Wikimedia project"@en-gb "property to indicate that an item is part of a group of items of particular interest for maintenance, management, or development."@en-gb
    # 'http://www.wikidata.org/prop/direct/P3411', # "Saxon Academy of Sciences member ID"@en "identifier in the members' database of the Saxon Academy of Sciences"@en
    # 'http://www.wikidata.org/prop/direct/P1331', # "PACE member ID"@en "identifier for a member of the Parliamentary Assembly of the Council of Europe"@en
    # 'http://www.wikidata.org/prop/direct/P2862', # "Catalogus Professorum Academiae Rheno-Traiectinae ID"@en "identifier for a professor at Utrecht University"@en
    # 'http://www.wikidata.org/prop/direct/P6231', # "BDEL ID"@en "identifier for a person on the 'Base de données des élites suisses'"@en
    # 'http://www.wikidata.org/prop/direct/P3368', # "Prabook ID"@en "identifier of a person in the Prabook database"@en
    'http://www.wikidata.org/prop/direct/P3373',
    # "sibling"@en-gb "the subject has the object as their sibling (brother, sister, etc.). Use "relative" (P1038) for siblings-in-law (brother-in-law, sister-in-law, etc.) and step-siblings (step-brothers, step-sisters, etc.)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P112', # "founder"@en-gb "the founder of this organisation, religion, or place"@en-gb
    # 'http://www.wikidata.org/prop/direct/P735', # "given name"@en-gb "first name or another given name of this person; values used with the property shouldn't link disambiguations nor family names."@en
    # 'http://www.wikidata.org/prop/direct/P6213', # "UK Parliament identifier"@en "identifier used by the UK Parliament linked-data system"@en
    # 'http://www.wikidata.org/prop/direct/P1153', # "Scopus Author ID"@en-gb "identifier for an author assigned in Scopus bibliographic database"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3901', # "ADAGP artist ID"@en "identifier for an artist as a member of the French collective rights management organisation ADAGP and sister organisations worldwide"@en
    # 'http://www.wikidata.org/prop/direct/P4854', # "Uppslagsverket Finland ID"@en "identifier of an article in the online encyclopedia Uppslagsverket Finland"@en
    # 'http://www.wikidata.org/prop/direct/P2605', # "ČSFD person ID"@en "identifier for a person in the Czech film database ČSFD"@en
    # 'http://www.wikidata.org/prop/direct/P2446', # "Transfermarkt player ID"@en "identifier for a association football (soccer) player, in the transfermarkt.com database"@en
    'http://www.wikidata.org/prop/direct/P97',  # "noble title"@en-gb "titles held by the person"@en-gb
    'http://www.wikidata.org/prop/direct/P2888',
    # "exact match"@en-gb "used to link two concepts, indicating a high degree of confidence that the concepts can be used interchangeably"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1015', # "NORAF ID"@en "identifier in the Norwegian information system BIBSYS"@en-gb
    # 'http://www.wikidata.org/prop/direct/P551', # "residence"@en-gb "the place where the person is, or has been, resident"@en-gb
    'http://www.wikidata.org/prop/direct/P2021',
    # "Erdős number"@en "the "collaborative distance" between mathematician Paul Erdős and another person. Use point in time (P585) as qualifier and should be used with a source."@en
    # 'http://www.wikidata.org/prop/direct/P7671', # "Semion author ID"@en "identifier of an author or reviewer, in Semion"@en
    # 'http://www.wikidata.org/prop/direct/P5443', # "Collège de France professor ID"@en-gb "identifier of a professor at the 'Collège de France' on its website"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2428', # "RePEc Short-ID"@en "identifier for a researcher in the RePEc (Research Papers in Economics) and IDEAS database"@en
    'http://www.wikidata.org/prop/direct/P737',
    # "influenced by"@en-gb "this person, idea, etc. is informed by that other person, idea, etc., e.g. "Heidegger was influenced by Aristotle"."@en
    # 'http://www.wikidata.org/prop/direct/P1146', # "World Athletics athlete ID"@en "identifier for athletes in World Athletics database and website"@en
    # 'http://www.wikidata.org/prop/direct/P1816', # "National Portrait Gallery (London) person ID"@en "identifier for sitters and artists represented in the National Portrait Gallery, London"@en
    # 'http://www.wikidata.org/prop/direct/P3630', # "Babelio author ID"@en "identifier for an author on the literature website Babelio"@en
    # 'http://www.wikidata.org/prop/direct/P1971' # "number of children"@en "number of children of the person. Mainly in cases where the full list isn't or shouldn't be added in P40."@en
    # 'http://www.wikidata.org/prop/direct/P3829', # "Publons author ID"@en-gb "identifier of an author or reviewer, in Publons"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3940', # "OlimpBase Chess Olympiad player ID"@en "identifier for a player at www.olimpbase.org who took part in the Chess Olympiad"@en
    # 'http://www.wikidata.org/prop/direct/P7859', # "WorldCat Identities ID"@en "detailed bibliographic info from WorldCat about an entity"@en
    # 'http://www.wikidata.org/prop/direct/P3258', # "LiveJournal ID"@en "username of a person or organisation, on LiveJournal"@en
    # 'http://www.wikidata.org/prop/direct/P6366', # "Microsoft Academic ID"@en-gb "identifier for an object in the Microsoft Academic Graph"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4521', # "Radio Radicale person ID"@en "identifier for a person in the database of Radio Radicale"@en
    'http://www.wikidata.org/prop/direct/P1066',  # "student of"@en-gb "person who has taught this person"@en-gb
    'http://www.wikidata.org/prop/direct/P6886', # "writing language"@en "language in which the writer has written their work"@en
    # 'http://www.wikidata.org/prop/direct/P691', # "NKCR AUT ID"@en-gb "identifier in the Czech National Authority Database (National Library of Czech Republic)"@en
    # 'http://www.wikidata.org/prop/direct/P4666', # "CineMagia person ID"@en "identifier for a person on the Cinemagia.ro website"@en
    # 'http://www.wikidata.org/prop/direct/P1430', # "OpenPlaques subject ID"@en-gb "identifier for a person or other subject in the OpenPlaques database - http://openplaques.org/"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2037', # "GitHub username"@en-gb "username of this project, person or organization on GitHub"@en
    # 'http://www.wikidata.org/prop/direct/P898', # "IPA transcription"@en-gb "transcription in the International Phonetic Alphabet"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5380', # "National Academy of Sciences member ID"@en-gb "identifier for a member or foreign associate on the American National Academy of Sciences website"@en-gb
    # 'http://www.wikidata.org/prop/direct/P366', # "use"@en-gb "main use of the subject (includes current and former usage)"@en-gb
    'http://www.wikidata.org/prop/direct/P1327',
    # "professional or sporting partner"@en-gb "professional collaborator"@en
    # 'http://www.wikidata.org/prop/direct/P2002', # "Twitter username"@en-gb "this item's username on Twitter; do not include the “@” symbol"@en
    # 'http://www.wikidata.org/prop/direct/P2287', # "CRIStin ID"@en "ID in the database for Norwegian scientists"@en
    # 'http://www.wikidata.org/prop/direct/P7578', # "DUC ID"@en "identifier for a noted woman on the online version of the ''Dictionnaire universel des créatrices''"@en
    # 'http://www.wikidata.org/prop/direct/P990', # "audio recording of the subject's spoken voice"@en "audio file representing the speaking voice of a person; or of an  animated cartoon or other fictitious character"@en
    # 'http://www.wikidata.org/prop/direct/P5819', # "International Mathematical Olympiad participant ID"@en "identifier of an International Mathematical Olympiad participant on the official website of IMO"@en
    'http://www.wikidata.org/prop/direct/P172',
    # "ethnic group"@en-gb "subject's ethnicity (consensus is that a VERY high standard of proof is needed for this field to be used. In general this means 1) the subject claims it him/herself, or 2) it is widely agreed on by scholars, or 3) is fictional and portrayed as such)."@en-gb
    # 'http://www.wikidata.org/prop/direct/P646', # "Freebase ID"@en-gb "identifier for a page in the Freebase database. Format: "/m/0" followed by 2 to 7 characters. For those starting with "/g/", use Google Knowledge Graph identifier (P2671)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P31', # "instance of"@en-gb "that class of which this subject is a particular example and member. (Subject typically an individual member with Proper Name label.) Different from P279 (subclass of)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P412', # "voice type"@en "person's voice type. expected values: soprano, mezzo-soprano, contralto, countertenor, tenor, baritone, bass (and derivatives)"@en
    'http://www.wikidata.org/prop/direct/P1026',
    # "doctoral thesis"@en-gb "thesis that someone wrote to obtain a PhD degree"@en-gb
    'http://www.wikidata.org/prop/direct/P103',
    # "native language"@en-gb "language or languages a person has learned from early childhood"@en
    'http://www.wikidata.org/prop/direct/P485',
    # "archives at"@en-gb "the institution holding the subject's archives"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2611', # "TED speaker ID"@en "identifier of a person, in the TED database of talks"@en
    # 'http://www.wikidata.org/prop/direct/P1695', # "NLP ID (unique)"@en "former National Library of Poland unique identifier. Format: "A", 7 digits, "X" or another digit. For the newer 16-digit format, use "NLP ID (PLWABN record)" (P7293)"@en
    # 'http://www.wikidata.org/prop/direct/P4252', # "All-Russian Mathematical Portal ID"@en "identifier for a mathematician in the All-Russian Mathematical Portal"@en
    # 'http://www.wikidata.org/prop/direct/P3762', # "openMLOL author ID"@en "identifier of an author in the openMLOL digital library of cultural resources"@en
    # 'http://www.wikidata.org/prop/direct/P2003',  # "Instagram username"@en-gb "item's username on Instagram"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P1741', # "GTAA ID"@en "identifier for GTAA, a thesaurus used in audiovisual archives (NISV, EYE)"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P244', # "Library of Congress authority ID"@en "only for authority control"@en-ca
    # 'http://www.wikidata.org/prop/direct/P4033',
    # "Mastodon address"@en "address on the Mastodon decentralized social network. The form is: 'user@server.domain' there is no leading '@' as sometimes written to distinguish Mastodon addresses from email addresses."@en
    # 'http://www.wikidata.org/prop/direct/P3747', # "SSRN author ID"@en "identifier for an author at the Social Science Research Network"@en
    # 'http://www.wikidata.org/prop/direct/P3280', # "BanQ author ID"@en "identifier for an author or subject heading in the Bibilothèque et Archives du Québec (BanQ)"@en
    # 'http://www.wikidata.org/prop/direct/P935', # "Commons gallery"@en-gb "name of the Wikimedia Commons gallery page(s) related to this item (is suitable to allow multiple links to more gallery pages)"@en
    # 'http://www.wikidata.org/prop/direct/P2416', # Sports discipline (too many relations)
    # "sports discipline competed in"@en "discipline an athlete competed in within a sport"@en
    # 'http://www.wikidata.org/prop/direct/P5212', # "Armenian National Academy of Sciences ID"@en "identifier for a member of the Armenian National Academy of Sciences"@en
    # 'http://www.wikidata.org/prop/direct/P1266', # "AlloCiné person ID"@en-gb "identifier for a person on the AlloCiné film database"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4598', # "FAPESP researcher ID"@en-gb "identifier for researchers funded by the Brazilian research education and innovation foundation, FAPESP"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1607', # "Dialnet author ID"@en "identifier of an author in Dialnet"@en
    # 'http://www.wikidata.org/prop/direct/P2070', # "Fellow of the Royal Society ID"@en-gb "Fellow ID of the Royal Society"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2163', # "FAST ID"@en-gb "authority control identifier in WorldCat's “FAST Linked Data” authority file"@en-gb
    # 'http://www.wikidata.org/prop/direct/P511',
    # "honorific prefix"@en "word or expression used before a name, in addressing or referring to a person"@en
    'http://www.wikidata.org/prop/direct/P2650',
    # "interested in"@en-gb "item of special or vested interest to this person or organisation"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1225',
    # "US National Archives Identifier"@en-gb "identifier for the United States National Archives and Records Administration's online catalog"@en-gb
    # 'http://www.wikidata.org/prop/direct/P553', # "website account on"@en-gb "website that the person or organization has an account on (use with P554) Note: only used with reliable source or if the person or organization disclosed it."@en
    # 'http://www.wikidata.org/prop/direct/P4629', # "Online Books Page author ID"@en "identifier for an author, at the Online Books Page website"@en
    # 'http://www.wikidata.org/prop/direct/P2191', # "NILF ID"@en "numeric identifier within the Vegetti Catalog of Fantastic Literature"@en
    # 'http://www.wikidata.org/prop/direct/P271', # "CiNii author ID (books)"@en-gb "identifier for a book author in CiNii (Scholarly and Academic Information Navigator)"@en
    # 'http://www.wikidata.org/prop/direct/P4174', # "Wikimedia username"@en-gb "user name of a person, across all Wikimedia projects; disclosing others' usernames violates the WMF privacy policy and may result in a block; see Wikidata:Oversight if you want to remove your information"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3191', # "IMIS person ID"@en "identifier for a person in IMIS, database of Flanders Marine Institute"@en
    # 'http://www.wikidata.org/prop/direct/P4012', # "Semantic Scholar author ID"@en-gb "identifier for an author in the Semantic Scholar database"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2170', # "Hansard (2006–March 2016) ID"@en "identifier for a British MP in Hansard (2006–March 2016)"@en
    # 'http://www.wikidata.org/prop/direct/P6594', # "Guggenheim fellows ID"@en-gb "identifier for a person awarded a Guggenheim fellowship in the arts"@en-gb
    # 'http://www.wikidata.org/prop/direct/P136', # "genre"@en-gb "genre of a creative work or genre in which an artist works"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5389', # "permanent resident of"@en "country or region that person have the legal status of permanent resident"@en
    # 'http://www.wikidata.org/prop/direct/P10', # "video"@en-gb "relevant video. For images, use the property P18. For film trailers, qualify with "object has role" (P3831)="trailer" (Q622550)"@en
    # 'http://www.wikidata.org/prop/direct/P1375', # "NSK ID"@en "identifier for an item in the National and University Library in Zagreb (including leading zeroes)"@en
    # 'http://www.wikidata.org/prop/direct/P4228', # "Encyclopedia of Australian Science ID"@en "identifier for a person or organisation in the Encyclopedia of Australian Science, an online compilation of biographical data about Australian scientists and their organisations"@en
    # 'http://www.wikidata.org/prop/direct/P3478', # "Songkick artist ID"@en-gb "identifier for an artist on Songkick"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2900', # "fax number"@en-gb "telephone number of fax line"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3956', # "National Academy of Medicine (France) Member ID"@en None
    # 'http://www.wikidata.org/prop/direct/P3052', # "Bloomberg person ID"@en "identifier for a business person, at Bloomberg"@en
    # 'http://www.wikidata.org/prop/direct/P6282', # "French Academy of Sciences member ID"@en "identifier for a member of the French Academy of Sciences on its website"@en
    # 'http://www.wikidata.org/prop/direct/P3159', # "UGentMemorialis ID"@en-gb "identifier for a professor at the Ghent University"@en-gb
    'http://www.wikidata.org/prop/direct/P1416',
    # "affiliation"@en-gb "organization that a person or organization is affiliated with (not necessary member of or employed by)"@en
    # 'http://www.wikidata.org/prop/direct/P950', # "Biblioteca Nacional de España ID"@en "identifier from the authority file of the Biblioteca Nacional de España. Format for persons: "XX" followed by 4 to 7 digits"@en
    # 'http://www.wikidata.org/prop/direct/P4232', # "Figshare author ID"@en "identifier for an author on Figshare"@en
    # 'http://www.wikidata.org/prop/direct/P5587', # "Libris-URI"@en "identifier for an item in the catalogue of the Swedish National Library"@en
    # 'http://www.wikidata.org/prop/direct/P5463', # "AE member ID"@en-gb "identifier for a member of the Academy of Europe"@en-gb
    'http://www.wikidata.org/prop/direct/P1035', # "honorific suffix"@en "word or expression with connotations conveying esteem or respect when used, after a name, in addressing or referring to a person"@en
    # 'http://www.wikidata.org/prop/direct/P2013', # "Facebook ID"@en-gb "official identifier for a person, product or organization in Facebook - everything after https://www.facebook.com/"@en
    # 'http://www.wikidata.org/prop/direct/P6264', # "Harvard Index of Botanists ID"@en "numerical identifier for a person in the Harvard Index of Botanists"@en
    # 'http://www.wikidata.org/prop/direct/P2949', # "WikiTree person ID"@en "identifier for an person in the WikiTree genealogy website"@en
    # 'http://www.wikidata.org/prop/direct/P4003', # "Facebook ID name/number"@en-gb "official Facebook page of this entity (only for use with URLs containing "/pages/")"@en-gb
    'http://www.wikidata.org/prop/direct/P1576',
    # "lifestyle"@en-gb "typical way of life of an individual, group, or culture"@en
    # 'http://www.wikidata.org/prop/direct/P5419', # "NYRB contributor ID"@en-gb "identifier for a contributor on the New York Review of Books website"@en-gb
    'http://www.wikidata.org/prop/direct/P570',  # "date of death"@en-gb "date on which the subject died"@en-gb
    # 'http://www.wikidata.org/prop/direct/P428', # "botanist author abbreviation"@en "standard form (official abbreviation) of a personal name for use in an author citation (only for names of algae, fungi and plants)"@en
    # 'http://www.wikidata.org/prop/direct/P3188', # "Nobel prize ID"@en-gb "ID in the Nobel prize organization homepage"@en
    # 'http://www.wikidata.org/prop/direct/P5705', # "LARB author ID"@en "identifier for an author on the Los Angeles Review of Books website"@en
    # 'http://www.wikidata.org/prop/direct/P268', # "BnF ID"@en-gb "identifier for the subject issued by BNF (Bibliothèque nationale de France). Format: 8 digits followed by a check-digit or letter, do not include the initial 'cb'."@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P906', # "SELIBR"@en-gb "identifier per National Library of Sweden Libris library catalog"@en
    # 'http://www.wikidata.org/prop/direct/P4293', # "PM20 folder ID"@en "identifier for a folder in the ZBW Pressemappe 20. Jahrhundert (20th century press archive)"@en
    # 'http://www.wikidata.org/prop/direct/P3874', # "Justia Patents inventor ID"@en "ID of an inventor in Justia Patents"@en
    # 'http://www.wikidata.org/prop/direct/P7704', # "Europeana entity"@en "Europeana entity id: for persons, places, topics"@en
    # 'http://www.wikidata.org/prop/direct/P102',
    # "member of political party"@en-gb "the political party of which this politician is or has been a member"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2581', # "BabelNet ID"@en-gb "ID in BabelNet encyclopedic dictionary"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4955', # "MR Author ID"@en-gb "Mathematical Reviews ID in MathSciNet"@en-gb
    # 'http://www.wikidata.org/prop/direct/P648',
    # "Open Library identifier"@en-gb "numeric identifier for works, editions and authors"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1050',
    # "medical condition"@en-gb "disease or other health problem affecting an individual human or other animal"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2080', # "AcademiaNet ID"@en-gb "identifier in the AcademiaNet database for excellent female scientists"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5785', # "EU Research participant ID"@en-gb "ID of organization in EU's Framework Programs for Research"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1233', # "ISFDB author ID"@en-gb "identifier for a person in the Internet Speculative Fiction Database"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P243', # "OCLC"@en-gb "identifier for a unique bibliographic record in OCLC WorldCat"@en
    # 'http://www.wikidata.org/prop/direct/P1065', # "archive URL"@en "URL to the archived web page specified with URL property"@en
    # 'http://www.wikidata.org/prop/direct/P2699', # "URL"@en-gb "location of a resource"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5731', # "Angelicum author ID"@en "identifier for an author in the online catalogue of the Pontifical University of Saint Thomas Aquinas"@en
    # 'http://www.wikidata.org/prop/direct/P3789', # "Telegram username"@en "this item's username, channel or group on Telegram"@en
    # 'http://www.wikidata.org/prop/direct/P3237', # "KU Leuven person ID"@en-gb "identifier for a person in the Who's Who database of the Catholic University of Leuven"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P5587', # "Libris-URI"@en "identifier for an item in the catalogue of the Swedish National Library"@en
    # 'http://www.wikidata.org/prop/direct/P2600', # "Geni.com profile ID"@en "profile on the Geni.com genealogy website"@en
    # 'http://www.wikidata.org/prop/direct/P1580', # "University of Barcelona authority ID"@en "register of authorities of the University of Barcelona"@en
    # 'http://www.wikidata.org/prop/direct/P6660', # "Rxivist author ID"@en "identifier for the author of scientific or academic preprint papers, in the Rxivist database"@en
    # 'http://www.wikidata.org/prop/direct/P214', # "VIAF ID"@en-gb "identifier for the Virtual International Authority File database [format: up to 22 digits]"@en
    # 'http://www.wikidata.org/prop/direct/P1309', # "EGAXA ID"@en "identifier in Bibliotheca Alexandrina"@en
    # 'http://www.wikidata.org/prop/direct/P1556', # "zbMATH author ID"@en "identifier of a person in the Zentralblatt MATH database"@en
    # 'http://www.wikidata.org/prop/direct/P3946', # "Dictionary Grierson ID"@en "ID for argentinian scientists and researchers from the Diccionario de científicos argentinos Dra. Grierson"@en
    # 'http://www.wikidata.org/prop/direct/P1284', # "Munzinger person ID"@en "identifier on the Munzinger Archiv"@en
    # 'http://www.wikidata.org/prop/direct/P7771', # "PersonalData.IO ID"@en "identifier for an entity in the wiki.personaldata.io database"@en
    # 'http://www.wikidata.org/prop/direct/P1368', # "LNB ID"@en "identifier assigned by the National Library of Latvia"@en
    # 'http://www.wikidata.org/prop/direct/P973', # "described at URL"@en-gb "item is described at the following URL"@en-gb
    # 'http://www.wikidata.org/prop/direct/P906', # "SELIBR"@en-gb "identifier per National Library of Sweden Libris library catalog"@en
    # 'http://www.wikidata.org/prop/direct/P1617', # "BBC Things ID"@en-gb "identifier in the BBC Things database"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P214', # "VIAF ID"@en-gb "identifier for the Virtual International Authority File database [format: up to 22 digits]"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P5034', # "National Library of Korea Identifier"@en "identifier for a person in the database of the National Library of Korea"@en
    # 'http://www.wikidata.org/prop/direct/P7709', # "ScienceOpen author ID"@en "unique identifier of an author in ScienceOpen database"@en
    # 'http://www.wikidata.org/prop/direct/P6634', # "LinkedIn personal profile ID"@en-gb "identifier for a person, on the LinkedIn website"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P1015', # "NORAF ID"@en "identifier in the Norwegian information system BIBSYS"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2962',
    # "title of chess person"@en "title awarded by a chess federation to a person"@en
    'http://www.wikidata.org/prop/direct/P27',
    # "country of citizenship"@en-gb "the object is a country that recognizes the subject as its citizen"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1340',  # "eye colour"@en-gb "colour of the irises of a person's eyes"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1303',  # "instrument"@en "musical instrument that a person plays"@en
    # 'http://www.wikidata.org/prop/direct/P586', # "IPNI author ID"@en "numerical identifier for a person in the International Plant Names Index"@en
    # 'http://www.wikidata.org/prop/direct/P2924', # "Great Russian Encyclopedia Online ID"@en "identifier for an entry on the official website of the Great Russian Encyclopedia"@en
    # 'http://www.wikidata.org/prop/direct/P106',
    # "occupation"@en-gb "occupation of a person; see also "field of work" (Property:P101), "position held" (Property:P39)"@en-gb
    #'http://www.wikidata.org/prop/direct/P710',
    # "participant"@en-gb "person, group of people or organization that actively takes/took part in the event"@en-gb
    # 'http://www.wikidata.org/prop/direct/P7040', # "NosDéputés.fr identifiant"@en "identifier for a french deputies in NosDéputés.fr"@en
    # 'http://www.wikidata.org/prop/direct/P1296', # "Gran Enciclopèdia Catalana ID"@en-gb "identifier for an item in the Gran Enciclopèdia Catalana"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1581', # "official blog"@en "URL to the blog of this person or organization"@en
    # 'http://www.wikidata.org/prop/direct/P2273', # "Heidelberg Academy for Sciences and Humanities member ID"@en "entry in the list of members of the Heidelberg Academy for Sciences and Humanities"@en
    # 'http://www.wikidata.org/prop/direct/P1006', # "NTA identifier (Netherlands)"@en-gb "identifier for persons (not:works) from the Dutch National Thesaurus for Author names (which also contains non-authors)"@en
    # 'http://www.wikidata.org/prop/direct/P1273', # "CANTIC"@en-gb "identifier for authority control managed by the National Library of Catalonia (BNC). Format: "a", 7 digits, "x" or digit."@en
    # 'http://www.wikidata.org/prop/direct/P3413', # "Leopoldina member ID"@en "identifier in the members' database of the Leopoldina – German Academy of Sciences"@en
    # 'http://www.wikidata.org/prop/direct/P1889', # "different from"@en-gb "item that is different from another item, with which it is often confused"@en-gb
    # 'http://www.wikidata.org/prop/direct/P6677', # "OpenEdition Books author ID"@en "identifier for an author on OpenEdition"@en
    # 'http://www.wikidata.org/prop/direct/P1440', # "FIDE player ID"@en "identifier on the FIDE database for chess players"@en
    # 'http://www.wikidata.org/prop/direct/P4369', # "Cairn author ID"@en-gb "identifier for an author in Cairn, an online library of French-language scholarly journals"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3065', # "RERO ID"@en "identifier in the Library network of Western Switzerland's RERO database"@en
    # 'http://www.wikidata.org/prop/direct/P1343', # "described by source"@en-gb "dictionary, encyclopaedia, etc. where this item is described"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3222', # "NE.se ID"@en "ID of article on the Swedish Nationalencyklopedin (NE.se) site"@en
    # 'http://www.wikidata.org/prop/direct/P4872', # "GEPRIS person ID"@en "Identifier of a person in GEPRIS database of funded research projects"@en
    # 'http://www.wikidata.org/prop/direct/P4491', # "Isidore ID"@en "identifier of a research on Isidore, a platform that collects links to scholarly documents and official texts"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P648', # "Open Library identifier"@en-gb "numeric identifier for works, editions and authors"@en-gb
    # 'http://www.wikidata.org/prop/direct/P227', # "GND identifier"@en-gb "international authority file of names, subjects, and organisations (please don't use type n = name, disambiguation)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4619', # "National Library of Brazil ID"@en "identifier for an element in the database of the National Library of Brazil"@en
    # 'http://www.wikidata.org/prop/direct/P641', # "sport"@en-gb "sport or discipline in which the entity participates or belongs to"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5370', # "Entomologists of the World ID"@en "identifier for an entry in the Biographies of the Entomologists of the World online database"@en
    # 'http://www.wikidata.org/prop/direct/P2190', # "C-SPAN person ID"@en "identifier for a person's appearances on C-SPAN"@en
    # 'http://www.wikidata.org/prop/direct/P2861', # "Leidse Hoogleraren ID"@en "identifier in the Leidse Hoogleraren, a catalogue of University Professors of Leiden University since 1575"@en
    # 'http://www.wikidata.org/prop/direct/P166', # "award received"@en-gb "awards received by a person, organisation or creative work"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2435', # "PORT person ID"@en "PORT-network film database: identifier for a person"@en
    # 'http://www.wikidata.org/prop/direct/P6275', # "copyright representative"@en "person or organisation who represents the copyright for this person or work of art"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P6366', # "Microsoft Academic ID"@en-gb "identifier for an object in the Microsoft Academic Graph"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4265', # "Reddit username"@en "username on the social news website Reddit"@en
    # 'http://www.wikidata.org/prop/direct/P4594', # "arXiv author ID"@en "identifier for an author on arXiv"@en
    # 'http://www.wikidata.org/prop/direct/P98', # "editor"@en-gb "editor of a compiled work such as a book or academic journal"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1651', # "YouTube video ID"@en "identifier of a video on YouTube; qualify trailers with "object has role" (P3831)="trailer" (Q622550). For channels use P2397; for playlists use P4300"@en
    # 'http://www.wikidata.org/prop/direct/P1048', # "NCL ID"@en "identifier for authority control issued by the National Central Library in Taiwan"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P5361', # "BNB person ID"@en-gb "identifier of a person in the British National Bibliography (bnb.data.bl.uk)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1532', # "country for sport"@en-gb "country a person or a team represents when playing a sport"@en-gb
    # 'http://www.wikidata.org/prop/direct/P18', # "image"@en-gb "image of relevant illustration of the subject; if available, use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image); only images which exist on Wikimedia Commons are acceptable"@en-gb
    'http://www.wikidata.org/prop/direct/P69',
    # "educated at"@en-gb "educational institution attended by the subject"@en-gb
    #'http://www.wikidata.org/prop/direct/P800', # "notable works"@en-gb "subject's notable scientific work or work of art"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1007', # "Lattes Platform number"@en "number for the Lattes Platform entry on the person or group. Relationship between entry and subject must be supported by a source"@en
    # 'http://www.wikidata.org/prop/direct/P1053', # "ResearcherID"@en-gb "identifier for a researcher in a system for scientific authors, primarily used in Web of Science"@en
    # 'http://www.wikidata.org/prop/direct/P4124', # "Who's Who in France biography ID"@en "unique identifier in the Who's Who in France online biography dictionary"@en
    # 'http://www.wikidata.org/prop/direct/P361', # "part of"@en-gb "subject is a part of that object. Inverse property of "has part" (P527)."@en-gb
    # 'http://www.wikidata.org/prop/direct/P7449', # "NARCIS researcher ID"@en "Dutch researchers with information about publications"@en
    # 'http://www.wikidata.org/prop/direct/P434', # "MusicBrainz artist ID"@en-gb "identifier for an artist in the MusicBrainz open music encyclopædia"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P5739', # "PUSC author ID"@en "identifier for an author in the online catalogue of the Pontifical University of the Holy Cross"@en
    'http://www.wikidata.org/prop/direct/P803', # "professorship"@en "professorship position held by this academic person"@en
    # 'http://www.wikidata.org/prop/direct/P2963', # "Goodreads author ID"@en-gb "identifier of an author, in the GoodReads.com website"@en-gb
    'http://www.wikidata.org/prop/direct/P569',  # "date of birth"@en-gb "date on which the subject was born"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3899', # "Medium username"@en "username of the Medium account of a person or an organization"@en
    # 'http://www.wikidata.org/prop/direct/P2671', # "Google Knowledge Graph identifier"@en-gb "identifier for Google Knowledge Graph API"@en-gb
    # 'http://www.wikidata.org/prop/direct/P349', # "National Diet Library Auth ID"@en "identifier for authority control per the National Diet Library of Japan"@en
    # 'http://www.wikidata.org/prop/direct/P5739', # "PUSC author ID"@en "identifier for an author in the online catalogue of the Pontifical University of the Holy Cross"@en
    # 'http://www.wikidata.org/prop/direct/P2799', # "BVMC person ID"@en "identifier of an author on the Biblioteca Virtual Miguel de Cervantes"@en
    # 'http://www.wikidata.org/prop/direct/P3417', # "Quora topic ID"@en-gb "identifier for a topic on Quora (English language version)"@en-gb
    'http://www.wikidata.org/prop/direct/P1038', # "relative"@en-gb "family member (qualify with "type of kinship", P1039; for direct family member please use specific property)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P496', # "ORCID"@en-ca "identifier for a person"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P7704', # "Europeana entity"@en "Europeana entity id: for persons, places, topics"@en
    # 'http://www.wikidata.org/prop/direct/P734',  # "family name"@en-gb "surname or last name of a person"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3885', # "History of Modern Biomedicine ID"@en "identifier of a person or topic in the History of Modern Biomedicine database"@en
    # 'http://www.wikidata.org/prop/direct/P1263', # "NNDB"@en-gb "identifier in the Notable Names Database, a biographical database: only for people entries"@en
    'http://www.wikidata.org/prop/direct/P185',  # "doctoral student"@en-gb "doctoral student(s) of a professor"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3468', # "National Inventors Hall of Fame ID"@en "identifier for an inductee in the United States National Inventors Hall of Fame"@en
    # 'http://www.wikidata.org/prop/direct/P1813', # "short name"@en-gb "short name of a place, organization, person etc."@en-ca
    # 'http://www.wikidata.org/prop/direct/P864', # "ACM Digital Library author ID"@en "Association for Computing Machinery Digital Library (ACM DL) author identifier"@en
    # 'http://www.wikidata.org/prop/direct/P2038', # "ResearchGate profile ID"@en "identifier for a person, used by ResearchGate"@en-gb
    # 'http://www.wikidata.org/prop/direct/P50', # Too many triples
    # "author"@en-gb "main creator(s) of a written work (use on works, not humans); use P2093 when Wikidata item is unknown or does not exist"@en
    # 'http://www.wikidata.org/prop/direct/P1890', # "BNC ID"@en "Biblioteca Nacional de Chile authority file ID"@en
    # 'http://www.wikidata.org/prop/direct/P396', # "SBN author ID"@en-gb "identifier issued by National Library Service (SBN) of Italy"@en
    # 'http://www.wikidata.org/prop/direct/P3602', # "candidacy in election"@en-gb "election where the subject is a candidate"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3721', # "public key fingerprint"@en "short sequence of bytes to identify a longer cryptographic public key"@en
    # 'http://www.wikidata.org/prop/direct/P2169', # "PublicWhip ID"@en "identifer in the PublicWhip database of British MPs"@en
    # 'http://www.wikidata.org/prop/direct/P2173', # "BBC News Democracy Live ID"@en "Identifer in the BBC News Democracy Live database of British MPs"@en
    'http://www.wikidata.org/prop/direct/P1087',
    # "Elo rating"@en "quantitative measure of one's game-playing ability, particularly in classical chess"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P2799', # "BVMC person ID"@en "identifier of an author on the Biblioteca Virtual Miguel de Cervantes"@en
    'http://www.wikidata.org/prop/direct/P22',
    # "father"@en-gb "male parent of the subject. For stepfather, use "stepparent' (P3448)"@en-ca
    # 'http://www.wikidata.org/prop/direct-normalized/P2163', # "FAST ID"@en-gb "authority control identifier in WorldCat's “FAST Linked Data” authority file"@en-gb
    'http://www.wikidata.org/prop/direct/P25',
    # "mother"@en-gb "female parent of the subject. For stepmother, use "stepparent" (P3448)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5552', # "CNRS talents page"@en "page on the Centre national de la recherche scientifique website presenting a person who has received certain prizes or medals"@en
    # 'http://www.wikidata.org/prop/direct/P2798', # "Loop ID"@en-gb "identifier for a person, in the Loop database of researcher impact"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2001', # "Revised Romanization"@en "romanisation following the Revised Romanisation of the Korean language"@en-gb
    # 'http://www.wikidata.org/prop/direct/P6479', # "IEEEXplore author ID"@en-gb "identifier for an author in IEEE Xplore"@en-gb
    'http://www.wikidata.org/prop/direct/P40',
    # "child"@en-gb "subject has the object in their family as their offspring son or daughter (independently of their age)"@en-ca
    'http://www.wikidata.org/prop/direct/P184',
    # "doctoral supervisor"@en-gb "person who supervised the doctorate or PhD thesis of the subject"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4285', # "Theses.fr person ID"@en-gb "identifier of a PhD holder or thesis jury member, on the French thesis database"@en-gb
    # 'http://www.wikidata.org/prop/direct/P7902', # "Deutsche Biographie ID"@en "identifier for a person in the Deutsche Biographie"@en
    # 'http://www.wikidata.org/prop/direct/P7882', # "ft.dk politician identifier"@en "identifier for a (former) member of the danish Folketing at ft.dk"@en
    # 'http://www.wikidata.org/prop/direct/P4450', # "HAL author ID"@en "identifier of a researcher on HAL, an open archive allowing to deposit scholarly documents freely searchable"@en
    # 'http://www.wikidata.org/prop/direct/P213', # "ISNI"@en-gb "International Standard Name Identifier for an identity. Format: 4 blocks of 4 digits separated by a space, first block is 0000"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1830',  # "owner of"@en-gb "entities owned by the subject"@en-gb
    'http://www.wikidata.org/prop/direct/P19', # "place of birth"@en-gb "most specific known (e.g. city instead of country, or hospital instead of city) birth location of a person, animal or fictional character"@en-gb
    # 'http://www.wikidata.org/prop/direct/P4081', # "BHL creator ID"@en "identifier for an author ("creator") in the Biodiversity Heritage Library database"@en
    # 'http://www.wikidata.org/prop/direct/P3029', # "UK National Archives ID"@en-gb "identifier for a person, family or organisation, in the UK's National Archives database"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P496',  # "ORCID"@en-ca "identifier for a person"@en
    # 'http://www.wikidata.org/prop/direct/P1417', # "Encyclopædia Britannica Online ID"@en-gb "identifer for an article in the online version of Encyclopædia Britannica"@en-gb
    'http://www.wikidata.org/prop/direct/P21',
    # "sex or gender"@en-gb "sexual identity of subject: male (Q6581097), female (Q6581072), intersex (Q1097630), transgender female (Q1052281), transgender male (Q2449503). Animals: male animal (Q44148), female animal (Q43445). Groups of same gender use "subclass of" (P279)"@en-gb
    #'http://www.wikidata.org/prop/direct/P585', # "point in time"@en-gb "time and date something took place, existed or a statement was true"@en-gb
    'http://www.wikidata.org/prop/direct/P451', # "unmarried partner"@en-gb "someone with whom the person is in a relationship without being married. Use "spouse" (P26) for married couples"@en-gb
    # 'http://www.wikidata.org/prop/direct/P53', # "family"@en-gb "family, including dynasty and nobility houses. Not family name (use P734 for family name)."@en
    # 'http://www.wikidata.org/prop/direct/P6656', # "BHCL ID"@en "identifier of authorities used in the Bibliography of the History of the Czech Lands (BHCL)"@en
    # 'http://www.wikidata.org/prop/direct/P243', # "OCLC"@en-gb "identifier for a unique bibliographic record in OCLC WorldCat"@en
    # 'http://www.wikidata.org/prop/direct/P4536', # "EThOS thesis ID"@en-gb "identifier of a doctoral thesis, in the British Library's EThOS database"@en
    # 'http://www.wikidata.org/prop/direct/P3708', # "PhDTree person ID"@en-gb "ID of a person at PhDTree"@en-gb
    # 'http://www.wikidata.org/prop/direct/P6178', # "Dimensions Author ID"@en-gb "unique identifier for an author in Dimensions"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3315', # "chesstempo ID"@en "identifier for chess players at chesstempo.com"@en
    # 'http://www.wikidata.org/prop/direct/P410', # "military rank"@en-gb "military rank achieved by a person (should usually have a "start time" qualifier), or military rank associated with a position"@en
    # 'http://www.wikidata.org/prop/direct/P2604', # "Kinopoisk person ID"@en "identifier for a person in the Kinopoisk.ru database"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P2671', # "Google Knowledge Graph identifier"@en-gb "identifier for Google Knowledge Graph API"@en-gb
    # 'http://www.wikidata.org/prop/direct/P2940', # "Catalogus Professorum Rostochiensium ID"@en "identifier in the Catalogus Professorum Rostochensium database on professors of the Rostock University from 1419 to present"@en
    # 'http://www.wikidata.org/prop/direct/P3277', # "KANTL member ID"@en "identifier for a member of the Royal Academy of Dutch language and literature"@en
    # 'http://www.wikidata.org/prop/direct/P1665', # "ChessGames.com player ID"@en "identifier on the website Chess Games (www.chessgames.com)"@en
    # 'http://www.wikidata.org/prop/direct/P968', # "e-mail"@en-gb "e-mail address of the organisation. Format: prefixed with mailto:"@en
    # 'http://www.wikidata.org/prop/direct/P512', # "academic degree"@en-gb "academic degree that the person holds"@en-ca
    # 'http://www.wikidata.org/prop/direct/P3314', # "365chess player ID"@en "identifier for players at 365chess.com"@en
    # 'http://www.wikidata.org/prop/direct/P2381', # "Academic Tree ID"@en-gb "identifer on academictree.org"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5034', # "National Library of Korea Identifier"@en "identifier for a person in the database of the National Library of Korea"@en
    # 'http://www.wikidata.org/prop/direct/P51', # "audio"@en-gb "relevant sound. If available, use a more specific property. Samples: "spoken text audio" (P989), "pronunciation audio" (P443)"@en
    # 'http://www.wikidata.org/prop/direct/P4123', # "French National Assembly ID"@en "identifier for a member of the French National Assembly on the official website (do not confuse with Sycomore ID)"@en
    # 'http://www.wikidata.org/prop/direct/P1207', # "NUKAT ID"@en-gb "identifier for authority control in the Center of Warsaw University Library catalogue"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P3348', # "National Library of Greece ID"@en "authority ID from the National Library of Greece Authority Records"@en
    # 'http://www.wikidata.org/prop/direct/P859', # "sponsor"@en-gb "organization or individual that sponsors this item"@en
    # 'http://www.wikidata.org/prop/direct/P2847', # "Google+ ID"@en-gb "Google+ account identifier of this person or organization: either starting with a "+" or consisting of 21 digits"@en-gb
    # 'http://www.wikidata.org/prop/direct/P6844', # "abART person ID"@en "identifier of person in the Czech database abART"@en
    # 'http://www.wikidata.org/prop/direct/P5715', # "Academia.edu profile URL"@en-gb "URL for a person's profile on the Academia.edu website"@en-gb
    # 'http://www.wikidata.org/prop/direct/P20', # "place of death"@en-gb "most specific known (e.g. city instead of country, or hospital instead of city) death location of a person, animal or fictional character"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P269', # "SUDOC identifier"@en-gb "identifier for authority control in the French collaborative library catalog (see also P1025). Format: 8 digits followed by a digit or "X""@en
    'http://www.wikidata.org/prop/direct/P937',  # "work location"@en-gb "location where persons were active"@en-gb
    'http://www.wikidata.org/prop/direct/P108', # "employer"@en-gb "person or organization for which the subject works or worked"@en
    # 'http://www.wikidata.org/prop/direct/P409', # "NLA"@en-gb "identifier issued by the National Library of Australia (see also P1315 for the newer People Australia identifier). VIAF component. Format: 1-12 digits, removing leading zero-padding."@en
    # 'http://www.wikidata.org/prop/direct/P2456', # "DBLP ID"@en-gb "identifier for person entries in the DBLP computer science bibliography (use portion of DBLP person key after homepages/)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5243', # "Canal-U person ID"@en "identifier of a person on Canal-U"@en
    # 'http://www.wikidata.org/prop/direct/P2893', # "Skype username"@en-gb "username on the Skype instant messaging service"@en-gb
    # 'http://www.wikidata.org/prop/direct-normalized/P4285', # "Theses.fr person ID"@en-gb "identifier of a PhD holder or thesis jury member, on the French thesis database"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3226', # "HAS member ID"@en "ID of the data-sheet of members of the Hungarian Academy of Sciences (Q265058)"@en
    'http://www.wikidata.org/prop/direct/P910',  # "topic's main category"@en-gb "main category of this topic"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1315', # "NLA Trove ID"@en "identifier for people per National Library of Australia (see also P409 for the older Libraries Australia identifier). Format: 6 to 8 digits."@en
    # 'http://www.wikidata.org/prop/direct/P3835', # "Mendeley person ID"@en-gb "identifier for an author of scholarly works, at mendeley.com"@en-gb
    # 'http://www.wikidata.org/prop/direct/P3233', # "PhilPeople profile"@en "an identifier for user profiles on PhilPeople"@en
    # 'http://www.wikidata.org/prop/direct/P4016', # "SlideShare username"@en "this item's username on SlideShare"@en
    # 'http://www.wikidata.org/prop/direct/P1280', # "CONOR identifier"@en-gb "identifier in the National and University Library, Ljubljana database"@en
    # 'http://www.wikidata.org/prop/direct/P4411', # "Quora username"@en "username of an individual or organisation, on the Quora website"@en
    # 'http://www.wikidata.org/prop/direct/P3987', # "SHARE Catalogue author ID"@en-gb "identifier for authors in SHARE Catalogue, a linked data federated catalogue of several Southern Italy universities"@en-gb
    # 'http://www.wikidata.org/prop/direct/P6551', # "Physics History Network ID"@en "identifier from the Physics History Network at the American Institute of Physics"@en
    # 'http://www.wikidata.org/prop/direct/P2174', # "Museum of Modern Art artist ID"@en "identifier assigned to an artist by the Museum of Modern Art"@en
    # 'http://www.wikidata.org/prop/direct/P2732', # "Persée author ID"@en "identifier for an author, in Persée"@en
    # 'http://www.wikidata.org/prop/direct/P1329', # "telephone number"@en-gb "telephone number in standard format (RFC3966), without `tel:` prefix"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1977', # "Les Archives du Spectacle Person ID"@en "Identifier for an actor/actress/playwright in the Les Archives du spectacle database"@en
    # 'http://www.wikidata.org/prop/direct-normalized/P646', # "Freebase ID"@en-gb "identifier for a page in the Freebase database. Format: "/m/0" followed by 2 to 7 characters. For those starting with "/g/", use Google Knowledge Graph identifier (P2671)"@en-gb

    # Wikidata incoming
    'http://www.wikidata.org/prop/direct/P802', # "student"@en-gb "notable student(s) of an individual"@en
    # 'http://www.wikidata.org/prop/direct/P488', # "chairperson"@en-gb "presiding member of an organization, group, or body"@en-ca
    # 'http://www.wikidata.org/prop/direct/P170', # "creator"@en-gb "maker of this creative work or other object (where no more specific property exists). Paintings with unknown painters, use "anonymous" (Q4233718) as value."@en
    # 'http://www.wikidata.org/prop/direct/P1889', # "different from"@en-gb "item that is different from another item, with which it is often confused"@en-gb
    # 'http://www.wikidata.org/prop/direct/P178', # "developer"@en-gb "organisation or person that developed this item"@en-gb
    # 'http://www.wikidata.org/prop/direct/P138', # "named after"@en-gb "person or other entity that inspired this item's name"@en-gb
    # 'http://www.wikidata.org/prop/direct/P710', # "participant"@en-gb "person, group of people or organization that actively takes/took part in the event"@en-gb
    'http://www.wikidata.org/prop/direct/P184', # "doctoral supervisor"@en-gb "person who supervised the doctorate or PhD thesis of the subject"@en-gb
    # 'http://www.wikidata.org/prop/direct/P112', # "founder"@en-gb "the founder of this organisation, religion, or place"@en-gb
    # 'http://www.wikidata.org/prop/direct/P5769', # "editor-in-chief"@en "a publication's editorial leader who has final responsibility for its operations and policies"@en
    'http://www.wikidata.org/prop/direct/P921', # "main subject"@en-gb "primary topic of a work (see also P180: depicts)"@en-gb
    'http://www.wikidata.org/prop/direct/P61', # "discoverer or inventor"@en-gb "the entity who discovered, first described,  invented, or developed this discovery or invention"@en
    'http://www.wikidata.org/prop/direct/P1066', # "student of"@en-gb "person who has taught this person"@en-gb
    # 'http://www.wikidata.org/prop/direct/P98', # "editor"@en-gb "editor of a compiled work such as a book or academic journal"@en-gb
    'http://www.wikidata.org/prop/direct/P3373', # "sibling"@en-gb "the subject has the object as their sibling (brother, sister, etc.). Use "relative" (P1038) for siblings-in-law (brother-in-law, sister-in-law, etc.) and step-siblings (step-brothers, step-sisters, etc.)"@en-gb
    # 'http://www.wikidata.org/prop/direct/P161', # "cast member"@en-gb "actor performing live for a camera or audience [use "character role" (P453) as qualifier] [use "voice actor" (P725) for voice-only role]"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1346', # "winner"@en-gb "winner of an event - do not use for awards (use P166/P2453 instead), nor for wars or battles"@en
    # 'http://www.wikidata.org/prop/direct/P5804', # "has program committee member"@en None
    'http://www.wikidata.org/prop/direct/P26', # "spouse"@en-gb "the subject has the object as their spouse (husband, wife, partner, etc.). Use "unmarried partner" (P451) for non-married companions"@en-gb
    'http://www.wikidata.org/prop/direct/P185', # "doctoral student"@en-gb "doctoral student(s) of a professor"@en-gb
    # 'http://www.wikidata.org/prop/direct/P1037', # "manager/director"@en-ca "person who manages any kind of group"@en-ca
    # 'http://www.wikidata.org/prop/direct/P823', # "speaker"@en "person who is speaker for this event, ceremony, keynote, presentation or in a literary work"@en
    # 'http://www.wikidata.org/prop/direct/P50', # "author"@en-gb "main creator(s) of a written work (use on works, not humans); use P2093 when Wikidata item is unknown or does not exist"@en
    # 'http://www.wikidata.org/prop/direct/P1855', # "Wikidata property example"@en-gb "example where this Wikidata property is used; target item is one that would use this property, with qualifier the property being described given the associated value"@en
    # 'http://www.wikidata.org/prop/direct/P126', # "maintenance"@en-ca "person or organization in charge of keeping the subject (for instance an infrastructure) in functioning order"@en
    'http://www.wikidata.org/prop/direct/P40', # "child"@en-gb "subject has the object in their family as their offspring son or daughter (independently of their age)"@en-ca
    # 'http://www.wikidata.org/prop/direct/P664', # "organizer"@en "person or institution organizing an event"@en-gb
    'http://www.wikidata.org/prop/direct/P22', # "father"@en-gb "male parent of the subject. For stepfather, use "stepparent' (P3448)"@en-ca
    # 'http://www.wikidata.org/prop/direct/P1327', # "professional or sporting partner"@en-gb "professional collaborator"@en
    # 'http://www.wikidata.org/prop/direct/P2554', # "production designer"@en "production designer(s) of this motion picture, play, video game or similar"@en
}

print(f'{len(whitelist)} relations whitelisted.')

# def get_wdentity(dblpid, wddoc):
#
#     dblp
#     triples, c = wddoc.search_triples('', 'http://www.wikidata.org/prop/direct/P2456', f'"{}}"')

def f(x : str):
    if x.startswith('_'): # blank node, leave as is
        return x
    if x.startswith('"'): # literal, rm newlines and escape internal quotes and slashes
        x = x[1:-1]
        x = x.replace('\n', '.newline').replace('\r', '.cr')
        lit = rdf.Literal(x)
        return lit.n3()

    else: # url, enclose with <>
        assert x.startswith('http') or x.startswith('file') or x.startswith('urn') or x.startswith('mailto'), x
        return f'<{x}>'

"""
Extract the DBLP data from the complete DBLP/Wikidata dumps.

We start with N DOIs for which we know the citation count (the target label).

We extract from DBLP the 2-hop neighborhood around the original paper to which the DOI belongs. This includes the author.

For each author that has an entry in Wikidata, we then retrieve the 1-hop neighborhood from Wikidata

"""

DBLP_DIR = '/Users/peter/Documents/datasets/dblp'
DBLP_FILE = 'dblp-20170124.hdt'

WD_DIR = '/Volumes/Port/wikidata/'
WD_FILE = 'wikidata20200309.hdt' # -- downloaded from the HDT website

# load the DBLP -> Wikidata mapping
authors = pd.read_csv('authors.csv', header=None)
authors.columns = ['dblp', 'wd']
authors.set_index('dblp')

authset = set(authors.dblp.values)

# Load the raw DBLP data
dblp = hdt.HDTDocument(DBLP_DIR + os.sep + DBLP_FILE)

doidf = pd.read_csv('citation_counts.csv', header=None)
instances = []
for doi in doidf[0]:
    triples, c = dblp.search_triples('', 'http://www.w3.org/2000/01/rdf-schema#seeAlso', 'http://dx.doi.org/' + doi)
    if c != 1:
        raise Exception(f'DOI {doi} has {c} associated publications in DBLP.')

    instance = next(triples)[0]
    instances.append(instance)

print(f'Retrieved {len(instances)} instances.')

print('Collecting 1 hop neighbors.')
include = set(instances) # the resources for which to include all neighboring triples
triples, c = dblp.search_triples('', '', '')

additions = set()
for s, p, o in tqdm.tqdm(triples, total=c):
    if s in include or o in include:
        additions.add(s)
        additions.add(o)

include.update(additions)

timing = Counter()
added  = Counter()

print('Writing 2 hop neighborhood to file.')
with gzip.open('dblp-reg.nt.gz', 'wt') as file:
    triples, c = dblp.search_triples('', '', '')
    for s, p, o in tqdm.tqdm(triples, total=c):
        if p in whitelist:
            if s in include or o in include:
                file.write(f'{f(s)} {f(p)} {f(o)} . \n')

    print('-- collecting wikidata additions')

    wd = hdt.HDTDocument(WD_DIR + os.sep + WD_FILE)
    print('   loaded wikidata')

    npeople, nptriples = 0, 0

    print('   matching orcids')
    for i, instance in enumerate(tqdm.tqdm(instances)):

        makertriples, cmake = dblp.search_triples(instance, 'http://xmlns.com/foaf/0.1/maker','')
        for _, _, maker in makertriples:

            if maker in authset:

                wdperson = authors.loc[authors.dblp == maker].wd.item()
                npeople += 1

                for p in whitelist:
                    if p.startswith('http://www.wikidata.org'):

                        tic()

                        persontriples, cperson = wd.search_triples(wdperson, p, '') # outgoing
                        for s, p, o in persontriples:
                            nptriples += 1
                            file.write(f'{f(maker)} {f(p)} {f(o)} . \n') # We tie the WD properties directly to the DBLP resource

                            added[p + ' out'] += 1

                        timing[p + ' out'] += toc()
                        tic()

                        persontriples, cperson = wd.search_triples('', p, wdperson) # incoming
                        for s, p, o in persontriples:
                            nptriples += 1
                            file.write(f'{f(s)} {f(p)} {f(maker)} . \n') # We tie the WD properties directly to the DBLP resource

                            added[p + ' in'] += 1
                        timing[p + ' in'] += toc()

        if i % 1000 == 0:
            for p, time in timing.most_common(5):
                print(f'{p}, {time:.4}')
            print()

            for p, num in added.most_common(5):
                print(f'{p}, {num}')
            print()

            print(f'Added {npeople} people from wikidata, with {nptriples} total triples. Checked {i} instances.')

print(f'Done. Added {npeople} people from wikidata, with {nptriples} total triples.')







