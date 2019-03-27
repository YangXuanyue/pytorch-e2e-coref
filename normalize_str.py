import re

S_PAT = re.compile(r'[’\']s ')
M_PAT = re.compile(r'[’\']m ')
VE_PAT = re.compile(r'[’\']ve ')
CANT_PAT = re.compile(r'can[’\']t')
NT_PAT = re.compile(r'n[’\']t ')
RE_PAT = re.compile(r'[’\']re ')
D_PAT = re.compile(r'[’\']d ')
LL_PAT = re.compile(r'[’\']ll ')
AT_PAT = re.compile(r'@')
HASH_PAT = re.compile(r'#')
PUNC_PAT = re.compile(r'([.,?!;:…][…<!"” \n])')
L_PUNC_PAT = re.compile(r'( *[({\["“])')
R_PUNC_PAT = re.compile(r'([)}\]"”]["” \n])')
SPACE_PAT = re.compile(r'\s{2,}')
DELIM_PAT = re.compile(r'[\n.?!…]')
# HTTP_PAT = re.compile(r'https*?://[^ \n]*')
# REF_PAT = re.compile(r'<ref[^>]*?>.*?</ref>')

# PUNCS = set('@#.,?!;:…[({\["“)}\]”')

# _LINK_PAT = re.compile(r'#\(([^#]+)\)#')
# _DISAMB_PAT = re.compile(r'_\([^)]+\)$')  # disambiguation
#
# _ENT_PAT = re.compile(r'#\(([^@]+)\)#')  # entity tag


def normalize_str(s):
    # s = REF_PAT.sub('', s)
    s = s.lower()
    s = M_PAT.sub(' \'m ', s)
    s = S_PAT.sub(' \'s ', s)
    s = VE_PAT.sub(' \'ve ', s)
    s = CANT_PAT.sub('cant', s)
    s = NT_PAT.sub(' n\'t ', s)
    s = RE_PAT.sub(' \'re ', s)
    s = D_PAT.sub(' \'d ', s)
    s = LL_PAT.sub(' \'ll ', s)
    s = AT_PAT.sub(' @ ', s)
    s = HASH_PAT.sub(' # ', s)
    s = PUNC_PAT.sub(r' \1 ', s)
    s = L_PUNC_PAT.sub(r' \1 ', s)
    s = R_PUNC_PAT.sub(r' \1 ', s)
    s = SPACE_PAT.sub(' ', s)
    return s.strip()





