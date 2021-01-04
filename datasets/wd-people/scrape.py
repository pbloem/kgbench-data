import hdt

doc = hdt.HDTDocument('/Volumes/Port/wikidata/wikidata20200309.hdt.gz')

print("nb triples: %i" % doc.total_triples)
print("nb subjects: %i" % doc.nb_subjects)
print("nb predicates: %i" % doc.nb_predicates)
print("nb objects: %i" % doc.nb_objects)
print("nb shared subject-object: %i" % doc.nb_shared)
