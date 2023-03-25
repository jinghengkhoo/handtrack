import math
import random

import pygame


def bresenham_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        points.reverse()
    return points

class Brush(pygame.sprite.Sprite):
    def __init__(self, pos, font, canvas, tmpcanvas, icon, brushes, offset):
        super().__init__(brushes)
        self.image = pygame.Surface((32, 32))
        self.image.fill(pygame.Color('grey'))
        font.render_to(self.image, (8, 7), icon)
        self.other_image = self.image.copy()
        pygame.draw.rect(self.other_image, pygame.Color('red'), self.other_image.get_rect(), 3)
        self.rect = self.image.get_rect(topleft=pos)
        self.active = False
        self.canvas = canvas
        self.tmpcanvas = tmpcanvas
        self.brushes = brushes
        self.offset = offset
        self.mouse_pos = None

    def translate(self, pos):
        return pos[0] - self.offset[0], pos[1] - self.offset[1]

    def draw_to_canvas(self):
        pass

    def flip(self):
        self.active = not self.active
        self.image, self.other_image = self.other_image, self.image

    def update(self, events, hand_loc=None):
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(e.pos):
                for brush in self.brushes:
                    if brush.active:
                        brush.flip()
                self.flip()
        if hand_loc:
            self.mouse_pos = self.translate(hand_loc)
        else:
            self.mouse_pos = self.translate(pygame.mouse.get_pos())
        if self.active:
            self.draw_to_canvas()

class Pencil(Brush):
    def __init__(self, pos, font, canvas, tmpcanvas, brushes, offset):
        super().__init__(pos, font, canvas, tmpcanvas, 'P', brushes, offset)
        self.prev_pos = None

    def draw_to_canvas(self):
        pressed = pygame.mouse.get_pressed()
        if pressed[0] and self.prev_pos:
            pygame.draw.line(self.canvas, pygame.Color('red'), self.prev_pos, self.mouse_pos)
        self.prev_pos = self.mouse_pos

class Calligraphy(Brush):
    def __init__(self, pos, font, canvas, tmpcanvas, brushes, offset):
        super().__init__(pos, font, canvas, tmpcanvas, 'C', brushes, offset)
        self.prev_pos = None

    def draw_to_canvas(self):
        if self.prev_pos:
            for x, y in bresenham_line(self.prev_pos, self.mouse_pos):
                pygame.draw.rect(self.canvas, pygame.Color('orange'), (x, y, 5, 15))
        self.prev_pos = self.mouse_pos

class Airbrush(Brush):
    def __init__(self, pos, font, canvas, tmpcanvas, brushes, offset):
        super().__init__(pos, font, canvas, tmpcanvas, 'A', brushes, offset)

    def draw_to_canvas(self):
        pygame.draw.circle(self.canvas, pygame.Color('green'),
        (self.mouse_pos[0] + random.randrange(-13, 13), self.mouse_pos[1] + random.randrange(-13, 13)),
        random.randrange(1, 5))

class FlowerBrush(Brush):
    def __init__(self, pos, font, canvas, tmpcanvas, brushes, offset):
        super().__init__(pos, font, canvas, tmpcanvas, 'F', brushes, offset)

    def draw_to_canvas(self):
        offset = (random.randrange(-5, 5), random.randrange(-5, 5))
        colors = ["#97C1A9", "#B7CFB7", "#CCE2CB", "#EAEAEA", "#C7DBDA", "#FFE1E9", "#FDD7C2", "#F6EAC2", "#FFB8B1", "#FFDAC1", "#E2F0CB", "#B5EAD6", "#55CBCD", "#A3E1DC", "#EDEAE5", "#FFDBCC", "#9AB7D3", "#F5D2D3", "#F7E1D3", "#DFCCF1"]
        center = random.randrange(5, 7)
        petals_num = random.randrange(6, 10, 2)
        angle = 0
        dangle = 2 * math.pi / petals_num
        petal_size = center = random.randrange(10, 15)
        petal_color = colors[random.randrange(0, 19)]
        for _ in range(petals_num):
            angle += dangle
            print((center * math.cos(angle), center * math.sin(angle)))
            pygame.draw.circle(
                self.canvas, pygame.Color(petal_color),
                (offset[0] + self.mouse_pos[0] + center * math.cos(angle), offset[1] + self.mouse_pos[1] + center * math.sin(angle)),
                petal_size
            )
        pygame.draw.circle(
            self.canvas, pygame.Color(colors[random.randrange(0, 19)]),
            (offset[0] + self.mouse_pos[0], offset[1] + self.mouse_pos[1]),
            center
            )
        # pygame.draw.circle(
        #     self.canvas, pygame.Color(colors[random.randrange(0, 19)]),
        #     (self.mouse_pos[0], self.mouse_pos[1]),
        #     center - 2
        #     )


class LineTool(Brush):
    def __init__(self, pos, font, canvas, tmpcanvas, brushes, offset):
        super().__init__(pos, font, canvas, tmpcanvas, 'L', brushes, offset)
        self.start = None

    def draw_to_canvas(self):
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:
            if not self.start:
                self.start = self.mouse_pos
            pygame.draw.line(self.tmpcanvas, pygame.Color('yellow'), self.start, self.mouse_pos)
        else:
            if self.start:
                pygame.draw.line(self.canvas, pygame.Color('yellow'), self.start, self.mouse_pos)
                self.start = None

class Clear(Brush):
    def __init__(self, pos, font, canvas, tmpcanvas, brushes, offset):
        super().__init__(pos, font, canvas, tmpcanvas, '*', brushes, offset)

    def draw_to_canvas(self):
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:
            self.canvas.fill((1, 1, 1))
            self.flip()