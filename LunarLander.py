import pygame
import numpy as np


class Rocket(pygame.sprite.Sprite):
    # Rocket images    
    filenames = ['sprite/rocket0.png', 'sprite/rocket1.png', 'sprite/rocket0l.png', 'sprite/rocket1l.png',
                 'sprite/rocket0r.png', 'sprite/rocket1r.png', 'sprite/rocket0lr.png', 'sprite/rocket1lr.png']
    rocketImages = [pygame.image.load(file) for file in filenames]
    
    # Dimensions of rocket
    height = rocketImages[0].get_height()
    width = rocketImages[0].get_width()
 
    def __init__(self):
        super().__init__()
        self.reset()

    @staticmethod
    def get_movement_variables(action):
        left = False
        boost = False
        right = False
        if action == 0:
            left = False
            boost = False
            right = False
        elif action == 1:
            left = False
            boost = True
            right = False
        elif action == 2:
            left = True
            boost = False
            right = False
        elif action == 3:
            left = False
            boost = False
            right = True
        return boost, left, right

    def reset(self):       
        # Fuel
        self.fuel = 100.0
        
        # Position of rocket
        self.y = 400
        self.x = 0 + 300*(np.random.rand()*2-1)

        # Speed
        self.xspeed = 0.0 + 3*(np.random.rand()*2-1)
        self.yspeed = 0.0 + 3*(np.random.rand()*2-1)
  
        # Booster off
        self.boost = False
        self.image = self.rocketImages[0]
        self.rect = self.image.get_rect()
     
        self.step((False, False, False))
 
    def step(self, action):
        # Unpack action
        self.boost, self.left, self.right = self.get_movement_variables(action)

        # Out of fuel, turn off rockets
        if self.fuel == 0:
            self.boost = self.left = self.right = False

        # Update rocket image
        self.image = self.rocketImages[self.boost*1+self.left*2+self.right*4]

        # Update position
        self.y -= self.yspeed*.1
        self.x += self.xspeed*.1
        self.rect.y = 600-self.y-self.height-15
        self.rect.x = 400+self.x-self.width/2
        
        # Update speeds
        if self.boost:
            self.yspeed -= 1.5
        else:
            self.yspeed += 1        
        if self.left:
            self.xspeed += 2
        if self.right:
            self.xspeed -= 2

        # Update fuel
        if self.boost:
            self.fuel -= .4
        if self.left:
            self.fuel -= .2
        if self.right:
            self.fuel -= .2
        self.fuel = max(0, self.fuel)

  
class LunarLander():
    # Fonts and colors
    textColor = (255, 255, 255) 
    goodColor = (0, 255, 0)
    badColor = (255, 0, 0)
    backgroundColor = (0, 0, 40)
    platformColor = (150, 150, 180)
    
    # Rendering?
    rendering = False

    def __init__(self):
        pygame.init()
        # Set up game
        self.sprites = pygame.sprite.Group()
        self.rocket = Rocket()
        self.sprites.add(self.rocket)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()
        self.max_vals = np.array([400, 600, 200, 140])
        self.observation_dim = len(self.max_vals)
        self.action_dim = 4
        self.reward = None
        self.reward_function = """
        distance_reward = - 1 * np.sqrt(state[0]**2 + state[1]**2)
        speed_reward = -1 * np.sqrt(state[2]**2 + state[3]**2)
        reward = distance_reward + speed_reward

        if self.game_over:
            if self.won:
                reward = 1000
            else:
                reward = -100"""

    def get_state(self):
        return np.array([self.rocket.x, self.rocket.y, self.rocket.xspeed, self.rocket.yspeed]) / self.max_vals

    def reset(self):
        # Program status
        self.game_over = False
        self.won = False
        self.rocket.reset()

    def step(self, action):
        # Step the rocket
        if not self.game_over:
            self.rocket.step(action)
        
        # Check if game is over
        if self.rocket.y <= 0 or self.rocket.y > 600 or abs(self.rocket.x) > 400:
            self.game_over = True
            # Criteria to win the game
            if self.rocket.yspeed <= 20 and abs(self.rocket.x) <= 20 and abs(self.rocket.xspeed) <= 20 and \
                    self.rocket.y <= 0:
                self.won = True

        state = self.get_state()

        distance_reward = - 1 * np.sqrt(state[0] ** 2 + state[1] ** 2)
        speed_reward = -1 * np.sqrt(state[2] ** 2 + state[3] ** 2)
        reward = distance_reward + speed_reward

        if self.game_over:
            if self.won:
                reward = 1000
            else:
                reward = -100

        self.reward = reward

        return state, reward, self.game_over


    def init_render(self):
        self.screen = pygame.display.set_mode([800, 600])
        pygame.display.set_caption('Lunar Lander')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
                
    def render(self):
        if not self.rendering:
            self.init_render()
            
        # Limit to 30 fps
        self.clock.tick(30)
     
        # Clear the screen
        self.screen.fill(self.backgroundColor)
        
        # Draw text
        values = [self.rocket.y, self.rocket.x, self.rocket.yspeed, self.rocket.xspeed, self.rocket.fuel, 0 if self.reward is None else self.reward]
        labels = ["Vertical distance", "Horizontal distance", "Vertical speed", "Horizontal speed", "Fuel", "reward"]
        colors = [self.textColor, 
                  self.goodColor if abs(self.rocket.x)<=20 else self.badColor, 
                  self.goodColor if self.rocket.yspeed<=20 else self.badColor, 
                  self.goodColor if abs(self.rocket.xspeed)<=20 else self.badColor, 
                  self.textColor,
                  self.textColor]
        for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
            text = self.font.render("{:}".format(round(val, 3)), True, col)
            self.screen.blit(text, (790-text.get_width(), 30*i))            
            text = self.font.render(lbl, True, self.textColor)
            self.screen.blit(text, (500, 30*i))            

        # Draw game over or you won       
        if self.game_over:
            if self.won:
                msg = 'Congratulations!'
                col = self.goodColor
            else:
                msg = 'Game over!'
                col = self.badColor
            text = self.font.render(msg, True, col)
            textpos = text.get_rect(centerx=self.background.get_width()/2)
            textpos.top = 300
            self.screen.blit(text, textpos)
            self.rocket.image = self.rocket.rocketImages[0]

        # Draw platform
        pygame.draw.rect(self.screen, self.platformColor, pygame.Rect(370,570,60,10))
        pygame.draw.rect(self.screen, self.platformColor, pygame.Rect(0,580,800,20))        
        
        # Draw sprites
        self.sprites.draw(self.screen)
     
        # Display
        pygame.display.flip()

    def close(self):
        pygame.quit()

 
boost = False        
left = False
right = False
