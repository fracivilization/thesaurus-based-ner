#!/bin/bash
make_opts=(
    "WITH_NC=False WITH_O=True CHUNKER=enumerated"
    "WITH_NC=False WITH_O=True CHUNKER=spacy_np"
    'NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200" WITH_O=False CHUNKER=spacy_np'
    'NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200" WITH_O=True CHUNKER=spacy_np'
    'NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200 GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour" WITH_O=False CHUNKER=spacy_np'
)

get_make_opts () {
    NEGATIVE_CATS=$1
    local make_opts=(
        "WITH_NC=False WITH_O=True CHUNKER=enumerated"
        "WITH_NC=False WITH_O=True CHUNKER=spacy_np"
        "NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=False CHUNKER=spacy_np"
        "NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=True CHUNKER=spacy_np"
        "NEGATIVE_CATS=\"${NEGATIVE_CATS} GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour\" WITH_O=False CHUNKER=spacy_np"
    )
    for line in ${make_opts[@]}; do
        echo $line
    done
}