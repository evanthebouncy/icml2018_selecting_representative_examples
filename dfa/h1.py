import random


class Node(object):

    def __init__(self, name, parent=None):
        self.parent = parent
        self.name = name
        self.left_child = None
        self.right_child = None
        self.true = 0
        self.false = 0
        self.dirty = False
        self.counts = (0, 0)
        self.index = None  # used to tell us which example it is from

    def go_left(self):
        if self.left_child is None:
            self.left_child = Node(self.name+'0', parent=self)
        self.dirty = True
        return self.left_child

    def go_right(self):
        if self.right_child is None:
            self.right_child = Node(self.name+'1', parent=self)
        self.dirty = True
        return self.right_child

    def increment(self, yes):
        if yes:
            self.true += 1
        else:
            self.false += 1

    def get_counts(self):
        if self.index is not None:
            return (self.true, self.false)
        if not self.dirty:
            return self.counts
        left_count = 0, 0
        right_count = 0, 0
        if self.left_child:
            left_count = self.left_child.get_counts()
        if self.right_child:
            right_count = self.right_child.get_counts()
        self.dirty = False
        c1 = self.true + left_count[0] + right_count[0]
        c2 = self.false + left_count[1] + right_count[1]
        self.counts = (c1, c2)
        
        return self.counts

    # TODO some metrics

class H1Tree(object):

    def __init__(self):
        self.root = Node('')
        # self.examples = []

    def add_example(self, s, yes, index):
        current = self.root
        for c in s:
            if c == '0':
                current = current.go_left()
            else:
                current = current.go_right()
        # make counter for true/false to make uncertainty metric?
        # check each level's t/f to see uncertainty... go down as low as needed to get an uncertainty (otherwise just grab one at random)
        current.increment(yes)
        # print 'Adding:', s
        # print current.name, current.counts
        current.index = index

    def get_top_examples(self):
        # calculate counts for all tree
        self.root.get_counts()
        examples = []

        nodes_to_check = [self.root]
        nodes_to_use = []

        # first check all possible nodes to see what needs to be used
        while len(nodes_to_check) > 0:
            current = nodes_to_check.pop(0)

            c1, c2 = current.get_counts()
            # print current.name, '({},{})'.format(c1, c2)
            if c1 == 0 or c2 == 0:
                nodes_to_use.append(current)
            else:
                if current.left_child:
                    nodes_to_check.append(current.left_child)
                if current.right_child:
                    nodes_to_check.append(current.right_child)
                

        # now get examples (indices)
        print 'derp', len(nodes_to_use)
        return map(self.get_example, nodes_to_use)


    def get_example(self, node):
        # gets a random example with this node as root
        if node.index is not None:
            return node.index
        else:
            nodes = [node.left_child, node.right_child]
            nodes = [n for n in nodes if n is not None]
            return self.get_example(random.choice(nodes))


if __name__ == '__main__':
    stuffs = [(0, '1100000100', False), (1, '1111000101', True), (2, '1001110010', False), (3, '0000001100', True), (4, '1111110001', True), (5, '0111001111', True), (6, '0100110111', True), (7, '1001101011', False), (8, '1001111000', False), (9, '0000100111', True), (10, '1111100110', False), (11, '1111011110', True), (12, '0100001001', True), (13, '0101000101', True), (14, '0000001000', True), (15, '0010111110', False), (16, '1101011001', True), (17, '1111010100', True), (18, '1101001111', True), (19, '1101001101', True), (20, '1111010010', False), (21, '1111110100', True), (22, '0011010110', False), (23, '1101001011', True), (24, '1011101010', False), (25, '0011010111', False), (26, '1101010111', False), (27, '1000101110', False), (28, '0000001111', True), (29, '0111110101', False), (30, '1111000111', True), (31, '0111100101', True), (32, '1110010010', False), (33, '1010011011', False), (34, '0101010000', True), (35, '1110001000', False), (36, '0010010010', False), (37, '0111011100', True), (38, '1011101001', False), (39, '0101111000', False), (40, '0101001101', True), (41, '1101111010', False), (42, '1110001100', False), (43, '1110111100', False), (44, '0001000000', True), (45, '0000001110', False), (46, '0111000100', True), (47, '0000100000', True), (48, '0010011010', False), (49, '0100101100', True), (50, '0000101010', True), (51, '0001010101', True), (52, '0001101100', True), (53, '0001101000', True), (54, '1101100111', True), (55, '1101010110', True), (56, '0100010111', True), (57, '0110101101', False), (58, '0110100011', False), (59, '1101111111', False), (60, '0100011110', True), (61, '1111110000', False), (62, '0100110101', True), (63, '0101110100', True), (64, '0101110011', True), (65, '1111011001', True), (66, '1001010101', False), (67, '0111100111', True), (68, '0001011100', False), (69, '0100010100', True), (70, '0111111100', True), (71, '0111010010', False), (72, '0011100000', False), (73, '0011111111', False), (74, '0100000010', True), (75, '1101000001', True), (76, '1001010011', False), (77, '0000101101', False), (78, '0001100001', True), (79, '0111111110', True), (80, '0100011011', True), (81, '1110110001', False), (82, '0100000000', True), (83, '0100100010', False), (84, '0011010011', False), (85, '0010001010', False), (86, '1100101001', False), (87, '1011101111', False), (88, '1111010000', True), (89, '1101110011', True), (90, '1011110000', False), (91, '0111001101', True), (92, '1100110000', False), (93, '0101011110', True), (94, '0111010011', True), (95, '1110101000', False), (96, '0101100101', True), (97, '0001001000', False), (98, '1100001010', True), (99, '1100101110', False), (100, '1111111100', True), (101, '1111111101', False), (102, '1010100101', False), (103, '0011111001', False), (104, '1011011000', False), (105, '1101110100', True), (106, '1000011011', False), (107, '1100001100', False), (108, '0001100101', False), (109, '0100010001', True), (110, '0100001101', False), (111, '0001100010', False), (112, '1111110110', True), (113, '0011111010', False), (114, '0010100101', False), (115, '0001110101', True), (116, '0101101111', False), (117, '1001000111', False), (118, '0000001011', True), (119, '0100100100', True), (120, '0110110110', False), (121, '0101000100', True), (122, '0111100010', True), (123, '1101110001', True), (124, '1111101000', False), (125, '1101111100', True), (126, '0001100100', True), (127, '0001011011', True), (128, '1001110101', False), (129, '0100011101', True), (130, '1000100011', False), (131, '0111101110', False), (132, '0100010110', True), (133, '0010100111', False), (134, '0010000111', False), (135, '1101111110', True), (136, '0101001010', False), (137, '1001000010', False), (138, '0111000101', True), (139, '1101011100', True), (140, '0111001001', False), (141, '1110010001', False), (142, '0001110001', False), (143, '1010101010', False), (144, '1111000010', True), (145, '0110001011', False), (146, '1011011100', False), (147, '0011011001', False), (148, '1000100101', False), (149, '0011010001', False), (150, '1000111000', False), (151, '0110010010', True), (152, '0111110000', True), (153, '0001000110', False), (154, '1100011001', False), (155, '1100010010', True), (156, '0000001101', True), (157, '0101001110', False), (158, '1010000011', False), (159, '0110100000', False), (160, '0001011101', True), (161, '1001001000', False), (162, '1111010011', True), (163, '1111110101', False), (164, '1001110100', False), (165, '0100111000', True), (166, '0010001100', False), (167, '0110001110', True), (168, '1111111000', False), (169, '1010001001', False), (170, '1010110111', False), (171, '1001000000', False), (172, '0100111011', True), (173, '0000100101', True), (174, '0111110100', True), (175, '0011000111', False), (176, '0110101011', False), (177, '1111010001', True), (178, '1101001100', True), (179, '0111101111', False), (180, '0010111001', False), (181, '0110011110', False), (182, '0101001000', True), (183, '0100111100', True), (184, '0010101000', False), (185, '0001111101', True), (186, '0011011000', False), (187, '1110111010', False), (188, '1111000001', False), (189, '0101100100', False), (190, '1110001011', False), (191, '0110110011', False), (192, '1100001110', True), (193, '1101101110', False), (194, '0111001011', True), (195, '1100010111', True), (196, '0101000000', False), (197, '1111000100', False), (198, '1011011110', False), (199, '0100110110', True), (200, '1000001001', False), (201, '0000000001', True), (202, '0100011111', True), (203, '0110111010', False), (204, '1001100001', False), (205, '1110101010', False), (206, '0110001010', True), (207, '0111011111', False), (208, '1111110011', False), (209, '0100110010', True), (210, '0100100111', False), (211, '0011011011', False), (212, '0001000100', False), (213, '1000110000', False), (214, '0011111011', False), (215, '0101000011', False), (216, '1000000101', False), (217, '0110000000', True), (218, '0101100010', True), (219, '0011101010', False), (220, '1001100110', False), (221, '0000100110', False), (222, '0111010000', True), (223, '1010101000', False), (224, '0100101011', True), (225, '0101000010', True), (226, '1100111001', False), (227, '1111011010', False), (228, '0111000010', True), (229, '0101011100', True), (230, '1101100101', True), (231, '0101011001', True), (232, '0101000110', True), (233, '1011010111', False), (234, '0000111110', True), (235, '1110100110', False), (236, '0010010001', False), (237, '0101010011', True), (238, '0001011010', True), (239, '1010111101', False), (240, '0000111001', True), (241, '1011111111', False), (242, '0000000110', True), (243, '1000010000', False), (244, '0100000101', True), (245, '1011110010', False), (246, '1101010001', True), (247, '0011101111', False), (248, '0001000101', False), (249, '0110011101', True), (250, '1101000110', True), (251, '0010011011', False), (252, '1000101000', False), (253, '1111011111', False), (254, '0000110010', False), (255, '0110101010', False), (256, '0101000111', True), (257, '0011001011', False), (258, '0010111000', False), (259, '0001101101', True), (260, '0100011000', True), (261, '1011010000', False), (262, '0111100011', False), (263, '0011001100', False), (264, '1100100010', False), (265, '0110110001', False), (266, '0000010100', False), (267, '0110011011', True), (268, '0001111111', True), (269, '0010000000', False), (270, '0111011110', True), (271, '0001110110', False), (272, '1010111111', False), (273, '0010001111', False), (274, '0000110001', False), (275, '0011111100', False), (276, '1111010101', False), (277, '0100001010', True), (278, '1001111001', False), (279, '0111101011', False), (280, '1101111001', True), (281, '0101010001', True), (282, '1101000101', True), (283, '0110100110', False), (284, '0101010110', True), (285, '0111010001', True), (286, '1100010101', True), (287, '0010101001', False), (288, '1011001011', False), (289, '1000100001', False), (290, '1101000111', True), (291, '1101111011', False), (292, '0110100111', False), (293, '0101010010', False), (294, '1001000011', False), (295, '1100110010', False), (296, '0111010111', False), (297, '0111000110', True), (298, '0111001000', True), (299, '1011011001', False), (300, '1110101001', False), (301, '0010111101', False), (302, '0110000101', False), (303, '1100011111', True), (304, '1110001010', False), (305, '0111010100', True), (306, '0101011000', False), (307, '0100010011', True), (308, '0011100011', False), (309, '0000000111', False), (310, '1000010110', False), (311, '1100111100', False), (312, '0110010001', False), (313, '1110010100', False), (314, '1101100001', False), (315, '0100010101', True), (316, '0011100101', False), (317, '1001010000', False), (318, '0010111011', False), (319, '1000000001', False), (320, '0110010101', True), (321, '0000000100', True), (322, '0001101001', True), (323, '0100101000', True), (324, '0011000100', False), (325, '1001010111', False), (326, '0111101101', False), (327, '1110101101', False), (328, '0101101011', False), (329, '0100000110', False), (330, '0010110101', False), (331, '1101010100', True), (332, '0001101110', True), (333, '1011000101', False), (334, '1010100010', False), (335, '0001111011', True), (336, '0010000100', False), (337, '1111111111', False), (338, '0001010100', False), (339, '0011101110', False), (340, '0100110100', True), (341, '0110100100', False), (342, '1100111011', False), (343, '1011001001', False), (344, '1101110000', True), (345, '1110101100', False), (346, '0100001100', False), (347, '0100000011', True), (348, '0101110001', True), (349, '0100000111', True), (350, '0011001111', False), (351, '1010010101', False), (352, '1001101001', False), (353, '0011000000', False), (354, '0010000101', False), (355, '0001101111', True), (356, '1100110101', False), (357, '1011101100', False), (358, '0001111100', False), (359, '0001010111', True), (360, '1001001001', False), (361, '1001111110', False), (362, '0111010110', True), (363, '1011100000', False), (364, '0100100001', True), (365, '1101100010', True)]
    tree = H1Tree()

    for (index, s, yes) in stuffs:
        tree.add_example(s, yes, index)

    examples = tree.get_top_examples()
    print len(stuffs), len(examples)
    print examples[0]


