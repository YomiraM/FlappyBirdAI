import pygame
import neat
import time
import os
import random

pygame.font.init()

WIN_WIDTH = 700
WIN_HEIGHT = 700

GEN = 0

birdImgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
pipeImg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
baseImg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
bgImg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
bgImg = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")),(WIN_WIDTH,WIN_HEIGHT))

STAT_FONT = pygame.font.SysFont("comicSans", 50)

class Bird:
    IMGS = birdImgs
    MAX_ROTATION = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x , y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y
    
    def move(self):

        self.tick_count += 1

        # Movement velocity

        displacement = self.velocity*self.tick_count + 1.5*self.tick_count**2

        if displacement >= 16:
            displacement = 16
        
        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        # Going upwards

        if displacement < 0 or self.y < self.height + 50: 
            if self.tilt < self.MAX_ROTATION: 
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt = self.ROTATION_VELOCITY

    def draw(self, window):
        self.img_count += 1

        # Bird image change

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]

        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]

        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]

        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]

        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # When the bird is falling it stops flapping

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2
        
        # Tilt the bird

        rotatedImage = pygame.transform.rotate(self.img, self.tilt)
        newRectangle = rotatedImage.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        window.blit(rotatedImage, newRectangle.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.pipeTop = pygame.transform.flip(pipeImg, False, True)
        self.pipeBottom = pipeImg

        self.passed = False
        self.setHeight()

    def setHeight(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.pipeTop.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, window):

        # Drawing top and bottom of the pipe 

        window.blit(self.pipeTop, (self.x, self.top))
        window.blit(self.pipeBottom, (self.x, self.bottom))

    def collide(self, bird):
        birdMask = bird.get_mask()
        topMask = pygame.mask.from_surface(self.pipeTop)
        bottomMask = pygame.mask.from_surface(self.pipeBottom)

        topOffset = (self.x - bird.x, self.top - round(bird.y))
        bottomOffset = (self.x - bird.x, self.bottom - round(bird.y))

        bottomPoint = birdMask.overlap(bottomMask, bottomOffset)
        topPoint = birdMask.overlap(topMask, topOffset)

        if topPoint or bottomPoint:
            return True
        
        return False

class Base:
    VELOCITY = 5
    WIDTH = baseImg.get_width()
    IMG = baseImg

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):

        # Making floor seem like it is scrolling

        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, window):

        # Drawing the floor 

        window.blit(self.IMG, (self.x1, self.y))
        window.blit(self.IMG, (self.x2, self.y))

def windowDraw(window,birds,pipes, base, score, GEN):
    window.blit(bgImg, (0,0))

    for pipe in pipes:
        pipe.draw(window)

    scoreLabel = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    window.blit(scoreLabel, (WIN_WIDTH - scoreLabel.get_width() - 15, 10))

    scoreLabel = STAT_FONT.render("Gen: " + str(GEN), 1, (255,255,255))
    window.blit(scoreLabel, (10, 10))

    scoreLabel = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    window.blit(scoreLabel, (10, 50))

    base.draw(window)

    for bird in birds:
        bird.draw(window)

    pygame.display.update()

def main(genomes, config):

    # Simulates current population of birds and set their fitness

    global GEN
    GEN += 1

    # List with the genome, neural object and bird object using that network

    nets = []
    ge = []
    birds = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        ge.append(genome)

    base = Base(630)
    pipes = [Pipe(600)]
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    clock = pygame.time.Clock()
    score = 0

    run = True

    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipeIndex = 0

        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipeTop.get_width():
                pipeIndex = 1

        else:
            run = False
            break

        # 0.1 of fitness for each frame that the bird stays alive

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1


        # Sends bird and pipes location to know if jump or not

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipeIndex].height), abs(bird.y - pipes[pipeIndex].bottom)))

            if output[0] > 0.5:
                bird.jump()
            
        addPipe = False
        remove = []

        for pipe in pipes:
            pipe.move()

            # Checks for collision

            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    addPipe = True

            if pipe.x + pipe.pipeTop.get_width() < 0:
                remove.append(pipe)

        if addPipe:
            score += 1

            for genome in ge:
                genome.fitness += 5

            pipes.append(Pipe(700))

        for r in remove:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 630 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
            
        if score > 50:
            pygame.quit()
            quit()

        base.move()
        windowDraw(window, birds, pipes, base, score, GEN)


def run (config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)