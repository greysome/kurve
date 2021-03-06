import pickle
import contextlib
from sys import argv
from time import sleep
import tensorflow as tf
from engine import Engine
from ai import AI
import config

# suppress welcome message
with contextlib.redirect_stdout(None):
    import pygame

GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 99, 71)
GREY = (192, 192, 192)
MAGENTA = (255, 0, 255)
PINK = (155, 155, 155)

class Game(Engine):
    def __init__(self, n_humans, n_ais, model, sector_theta, fov):
        super().__init__(n_humans+n_ais,
                         w=500, h=500,
                         sector_theta=sector_theta, fov=fov)

        self.n_humans = n_humans
        self.n_ais = n_ais

        # surface to draw the body (coloured circles)
        self.tmp_sfc = None
        # surface to draw the gaps in body (black circles)
        self.body_sfc = None

        self.started = False
        self.pause = False

        self.player_keys = {}
        self._add_player_keys(0, pygame.K_LEFT, pygame.K_RIGHT)
        self._add_player_keys(1, pygame.K_1, pygame.K_2)
        self._add_player_keys(2, pygame.K_g, pygame.K_h)
        self._add_player_keys(3, pygame.K_o, pygame.K_p)
        self._add_player_keys(4, pygame.K_5, pygame.K_6)
        self.player_colours = {0: GREEN, 1: BLUE, 2: RED,
                               3: YELLOW, 4: WHITE, 5: ORANGE,
                               6: GREY, 7: MAGENTA, 8: PINK}

        self.ai_ids = range(n_humans, n_humans+n_ais)
        self.ai_states = {}
        self.ai = AI(n_inputs=config.fov//config.sector_theta,
                     n_actions=3,
                     hidden_neurons=config.hidden_neurons,
                     learning_rate=0)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        with open(model+'.model', 'rb') as f:
            W1, W2 = pickle.load(f)
            W1_assign = self.ai.W1.assign(W1)
            W2_assign = self.ai.W2.assign(W2)
            self.sess.run((W1_assign, W2_assign))

        self._pygame_init()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    if not self.started:
                        self.started = True
                    if e.key == pygame.K_q:
                        self._cleanup()
                        return
                    if e.key == pygame.K_m:
                        self.pause = not self.pause

            if self.pause or not self.started:
                continue

            ended, survivor = self.game_ended()
            if ended:
                if survivor:
                    print(f'{survivor.id_} wins!')
                else:
                    print('nobody wins!')
                self._cleanup()
                return

            self._draw_players_preupdate()
            self.step(self._get_human_updates() + self._get_ai_updates())
            self.ai_states = {}
            for id_ in self.ai_ids:
                state = self.observe(id_)
                self.ai_states.update({id_: state})
            self._draw_players_postupdate()

            self.screen.blit(self.tmp_sfc, (0, 0))
            self.screen.blit(self.body_sfc, (0, 0))
            pygame.display.flip()
            sleep(0.03)

    def reset(self):
        super().reset()
        self.tmp_sfc.fill(BLACK)
        self.body_sfc.fill(BLACK)
        self.screen.blit(self.tmp_sfc, (0, 0))
        self.screen.blit(self.body_sfc, (0, 0))

    def _cleanup(self):
        self.sess.close()

    def _is_human(self, id_):
        # only the first `n_humans` ids are for humans
        return id_ < self.n_humans

    def _add_player_keys(self, id_, key_left, key_right):
        if not self._is_human(id_):
            return

        player_keys = {key_left: {'id': id_,
                                  'direction': Game.ACTION_LEFT},
                       key_right: {'id': id_,
                                   'direction': Game.ACTION_RIGHT}}
        self.player_keys.update(player_keys)

    def _pygame_init(self):
        pygame.init()
        pygame.display.set_caption('kurve')
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.body_sfc = pygame.Surface((self.w, self.h),
                                       flags=pygame.SRCALPHA)
        self.tmp_sfc = pygame.Surface((self.w, self.h),
                                      flags=pygame.SRCALPHA)

    def _get_human_updates(self):
        updates = []
        pressed_keys = pygame.key.get_pressed()
        for k, key_info in self.player_keys.items():
            if pressed_keys[k]:
                p = self._find_player(key_info['id'])
                if not p.is_alive:
                    continue
                updates.append((key_info['id'], key_info['direction']))
        return updates

    def _get_ai_updates(self):
        updates = []
        for id_, inputs in self.ai_states.items():
            action = self.sess.run(self.ai.action,
                                   feed_dict={self.ai.inputs: inputs})
            updates.append((id_, action))
        return updates

    def _draw_players_preupdate(self):
        for p in self.players:
            if p.is_drawing_gap():
                # 'erase' the previous head
                pygame.draw.circle(self.tmp_sfc, BLACK,
                                   (round(p.x), round(p.y)), p.r)

    def _draw_players_postupdate(self):
        for p in self.players:
            pygame.draw.circle(
                (self.tmp_sfc if p.is_drawing_gap() else self.body_sfc),
                self.player_colours[p.id_],
                (round(p.x), round(p.y)),
                p.r
            )
      
if __name__ == "__main__":
    try:
        n_humans = int(argv[1])
        n_ais = int(argv[2])
        model = (argv[3] if len(argv) >= 4 else 'trained')
    except:
        print('usage: python game.py <n_humans> <n_ais> <custom model>')
        exit()
    else:
        if not 1 <= n_humans+n_ais <= 3:
            print('total number of players between 1 and 3')
            exit()

    game = Game(n_humans=n_humans, n_ais=n_ais, model=model,
                sector_theta=config.sector_theta, fov=config.fov)
    game.run()
