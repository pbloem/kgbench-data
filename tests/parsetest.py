
import unittest
import kgbench as kg


class TestUtil(unittest.TestCase):

    def test_parse(self):

        term = kg.parse_term('<http://kgbench.info/dt#base64Image>')
        self.assertEqual(type(term), kg.IRIRef)
        self.assertEqual(term.value, 'http://kgbench.info/dt#base64Image')
        self.assertEqual(term.n3(), '<http://kgbench.info/dt#base64Image>')

        term = kg.parse_term('_:alice')
        self.assertEqual(type(term), kg.BNode)
        self.assertEqual(term.value, 'alice')
        self.assertEqual(term.n3(), '_:alice')

        term = kg.parse_term('"Simple string."')
        self.assertEqual(type(term), kg.Literal)
        self.assertEqual(term.value, 'Simple string.')
        self.assertIsNone(term.language)
        self.assertIsNone(term.datatype)
        self.assertEqual(term.value, 'Simple string.')
        self.assertEqual(term.n3(), '"Simple string."')

        term = kg.parse_term('""@Dothraki')
        self.assertIsNone(term.datatype)
        self.assertEqual(term.language, "Dothraki")
        self.assertEqual(term.value, '')
        self.assertEqual(term.n3(), '""@Dothraki')

        term = kg.parse_term('"Cette Série des Années Septante"  @fr-be')
        self.assertIsNone(term.datatype)
        self.assertEqual(term.language, "fr-be")
        self.assertEqual(term.value, 'Cette Série des Années Septante')
        self.assertEqual(term.n3(), '"Cette Série des Années Septante"@fr-be')

        term = kg.parse_term('"1.663E-4"  ^^<http://www.w3.org/2001/XMLSchema#double>')
        self.assertIsNone(term.language)
        self.assertEqual(term.datatype.value, "http://www.w3.org/2001/XMLSchema#double")
        self.assertEqual(term.value, '1.663E-4')
        self.assertEqual(term.n3(), '"1.663E-4"^^<http://www.w3.org/2001/XMLSchema#double>')

        term = kg.parse_term('\t   "This is a multi-line\nliteral with many quotes (\\"\\"\\"\\"\\")\nand two apostrophes (\'\')."  ')
        self.assertIsNone(term.language)
        self.assertIsNone(term.datatype)
        self.assertEqual(term.value, 'This is a multi-line\nliteral with many quotes (""""")\nand two apostrophes (\'\').')
        # -- note the un-escaped quotes
        self.assertEqual(term.n3(), '"This is a multi-line\nliteral with many quotes (\\"\\"\\"\\"\\")\nand two apostrophes (\'\')."')

    def test_hdt_parse(self):

        term = kg.n3('http://kgbench.info/dt#base64Image')
        self.assertEqual(term, '<http://kgbench.info/dt#base64Image>')

        term = kg.n3('_:alice')
        self.assertEqual(term, '_:alice')

        term = kg.n3('"Simple string."')
        self.assertEqual(term, '"Simple string."')

        term = kg.n3('""@Dothraki')
        self.assertEqual(term, '""@Dothraki')

        term = kg.n3('"Cette Série des Années Septante"@fr-be')
        self.assertEqual(term, '"Cette Série des Années Septante"@fr-be')

        term = kg.n3('"1.663E-4"^^<http://www.w3.org/2001/XMLSchema#double>')
        self.assertEqual(term, '"1.663E-4"^^<http://www.w3.org/2001/XMLSchema#double>')

        term = kg.n3(
            '"This is a multi-line\nliteral with many quotes (""""")\nand two apostrophes (\'\')."', escape=False)
        self.assertEqual(term,
                         '"This is a multi-line\nliteral with many quotes (\\"\\"\\"\\"\\")\nand two apostrophes (\'\')."')
