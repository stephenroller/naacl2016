#!/usr/bin/env python
import sys
import unidecode
from xml.dom.minidom import parse

dom = parse(sys.argv[1])

for n, node in enumerate(dom.getElementsByTagName("lexelt")):
    # todo check this
    item = node.attributes['item'].nodeValue

    for i, subnode in enumerate(node.getElementsByTagName("instance")):
        sentenceno = int(subnode.attributes['id'].nodeValue)
        contextNode = subnode.childNodes[1]
        if len(contextNode.childNodes) != 3:
            assert len(contextNode.childNodes) == 2
            if contextNode.childNodes[0].nodeName == "head":
                left = ""
                head = contextNode.childNodes[0].childNodes[0].nodeValue
                right = contextNode.childNodes[1].nodeValue
            elif contextNode.childNodes[1].nodeName == "head":
                left = contextNode.childNodes[0].nodeValue
                head = contextNode.childNodes[1].childNodes[0].nodeValue
                right = ""
            else:
                raise ValueError
        else:
            leftNode, headNode, rightNode = contextNode.childNodes
            left = leftNode.nodeValue.strip()
            head = headNode.childNodes[0].nodeValue.strip()
            right = rightNode.nodeValue.strip()

        sentence = unidecode.unidecode(left + " " + head + " " + right)
        index = len(left.split())

        print "%s\t%d\t%d\t%s" % (unidecode.unidecode(item), sentenceno, index, sentence)


