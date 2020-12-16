"""
Some simple utilities for parsing RDF in ntriples. Adapted from

https://gitlab.com/wxwilcke/pyRDF

"""


def parse_statement(statement):
    statement = statement.rstrip(' ')
    if not statement.endswith('.'):
        statement = strip_comment(statement)

    statement = statement.rstrip(' .')

    subject, remainder = parse_subject(statement)
    predicate, remainder = parse_predicate(remainder)
    object = parse_term(remainder)

    return subject, predicate, object


def strip_comment(statement):
    for i in range(1, len(statement)):
        if statement[-i] == '#':
            break

    return statement[:-i]


def parse_subject(self, statement):
    if statement.startswith("_:"):
        return self._parse_bnode(statement)
    else:  # iriref
        return self._parse_iriref(statement)


def parse_predicate(self, statement):
    return self._parse_iriref(statement)


def parse_term( statement : str):
    """
    :param statement:
    :return:
    """

    statement = statement.strip()

    if statement.startswith('<'):
        object, _ = parse_iriref(statement)
        return object

    elif statement.startswith("_:"):
        object, _ = parse_bnode(statement)
        return object

    elif statement.startswith('"'):
        object = parse_literal(statement)
        return object
    else:
        raise Exception("Unexpected format: " + statement)

def parse_literal(statement):
    statement = statement.strip()

    qstart = 0
    qend = eat_quoted(qstart, statement)

    quoted = statement[qstart+1 : qend]
    quoted = decode(quoted)
    # -- decode any special characters

    i = eat_whitespace(qend+1, statement)

    if i >= len(statement): # no language tag, no datatype
        return Literal(quoted)

    if statement[i] == '@':
        remainder = statement[i+1:].strip()
        assert not any([r.isspace() for r in remainder]), f'Whitespace in language tag: {statement}'

        return Literal(quoted, language=remainder)

    if statement[i:i+2] == '^^':
        remainder = statement[i+2:].strip()
        assert remainder[0] == '<' and remainder[-1] == '>'
        datatype = remainder[1:-1]

        return Literal(quoted, datatype=datatype)

    raise Exception(f'Could not parse literal: {statement}')

def eat_quoted(i, string):
    """
    :param i: Index of the first quote mark
    :param string:
    :return: Index of the end of the closing quote mark
    """

    assert string[i] == '"'

    i += 1
    while string[i] != '"':
        if string[i:i+2] == r'\"':
            i += 2
        else:
            i += 1

        if i >= len(string):
            raise Exception(f'Could not parse {string}.')

    return i

def eat_whitespace(i, string):
    while i < len(string) and string[i].isspace():
        i += 1

    return i

def parse_bnode(statement):
    entity, remainder = parse_entity(statement)
    bnode = entity.value
    if bnode.startswith('_:'):
        bnode = BNode(bnode[2:])
    else:
        raise Exception("Unexpected format: " + bnode)

    return (bnode, remainder)


def parse_iriref(statement):
    entity, remainder = parse_entity(statement)

    iriref = entity.value
    if iriref.startswith('<'):
        iriref = IRIRef(iriref[1:-1])
    else:
        raise Exception("Unexpected format: " + iriref)

    return (iriref, remainder)


def parse_entity(statement):
    i = 0
    while i < len(statement) and statement[i] not in [u'\u0009', u'\u0020']:
        i += 1

    return (Entity(statement[:i]), statement[i + 1:].lstrip())

class Resource:
    value = None

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def __hash__(self):
        return hash(repr(self))

    def n3(self):
        pass

class Entity(Resource):
    def __init__(self, value):
        super().__init__(value)

class Literal(Resource):
    datatype = None
    """
    The literal's language tag.
    """

    language = None
    """
    The literal's datatype. Note that this is an IRIRef object if not None.
    """

    def __init__(self, value, datatype=None, language=None):
        """
        NB: Literal.value does not have it's double quote characters escaped. The n3() method escapes the quotes.

        :param value: Body of the string.
        :param datatype:
        :param language:
        """
        super().__init__(value)

        if datatype is not None and language is not None:
            raise Warning("Accepts either datatype or language, not both")

        self.datatype = IRIRef(datatype) if datatype is not None else None
        self.language = language

    def __eq__(self, other):
        return self.value == other.value\
                and self.datatype == other.datatype\
                and self.language == other.language

    def __hash__(self):
        value = str()
        if self.datatype is not None:
            value = self.datatype
        if self.language is not None:
            value = self.language

        return hash(repr(self)+repr(value))

    def n3(self):
        # literal
        body = encode(self.value)

        res = '"' + body + '"'
        if self.language is not None:
            res += '@' + self.language
        elif self.datatype is not None:
            res += '^^' + self.datatype.n3()

        return res

class BNode(Entity):
    def __init__(self, value):
        super().__init__(value)

    def n3(self):
        return '_:' + self.value

class IRIRef(Entity):
    def __init__(self, value):
        super().__init__(value)

    def n3(self):
        return '<' + self.value + '>'

def encode(s : str):
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    return s

def decode(s : str):
    s = s.replace('\\"', '"')
    s = s.replace('\\\\', '\\')
    return s
