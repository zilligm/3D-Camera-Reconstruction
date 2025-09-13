
class ObjectMapping():
    def __init__(self, object_type: str):

        if object_type == "body":
            self.point_mapping = {
                1: ('Nose', 'g'),
                2: ('Left Eye', 'g'),
                3: ('Right Eye', 'g'),
                4: ('Left Ear', 'g'),
                5: ('Right Ear', 'g'),
                6: ('Left Shoulder', 'b'),
                7: ('Right Shoulder', 'b'),
                8: ('Left Elbow', 'b'),
                9: ('Right Elbow', 'b'),
                10: ('Left Wrist', 'b'),
                11: ('Right Wrist', 'b'),
                12: ('Left Hip', 'r'),
                13: ('Right Hip', 'r'),
                14: ('Left Knee', 'r'),
                15: ('Right Knee', 'r'),
                16: ('Left Ankle', 'r'),
                17: ('Right Ankle', 'r'),
            }

            self.junctions = [(0, 1), (0, 2), (0, 3), (2, 4), (0, 3), (3, 4),
                              (2, 6), (3, 5), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                              (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

            self.colors = [x[1] for x in self.point_mapping.values()]

        elif object_type == "hand":
            self.point_mapping = {
                1: ('Wrist', 'k'),
                2: ('Thumb CMC', 'g'),
                3: ('Thumb MCP', 'g'),
                4: ('Thumb IP', 'g'),
                5: ('Thumb TIP', 'g'),
                6: ('Index CMC', 'b'),
                7: ('Index PIP', 'b'),
                8: ('Index DIP', 'b'),
                9: ('Index TIP', 'b'),
                10: ('Middle CMC', 'r'),
                11: ('Middle PIP', 'r'),
                12: ('Middle DIP', 'r'),
                13: ('Middle TIP', 'r'),
                14: ('Ring CMC', 'g'),
                15: ('Ring PIP', 'g'),
                16: ('Ring DIP', 'g'),
                17: ('Ring TIP', 'g'),
                18: ('Little CMC', 'r'),
                19: ('Little PIP', 'r'),
                20: ('Little DIP', 'r'),
                21: ('Little TIP', 'r'),
            }

            self.junctions = [(4, 3), (3, 2), (2, 1), (1, 0), (8, 7), (7, 6),
                              (6, 5), (5, 0), (5, 9), (12, 11), (11, 10), (10, 9), (9, 13),
                              (16, 15), (15, 14), (14, 13), (13, 17), (17, 18), (18, 19), (19, 20)]

            self.colors = [x[1] for x in self.point_mapping.values()]

        else:
            self.point_mapping = {}
            self.junctions = []
            self.colors = 'k'

        self.color_mapping = {'g': (255, 0, 0), 'b': (0, 255, 0), 'r': (0, 0, 255)}
