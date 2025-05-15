

# Aprendizaje por Refuerzo: Fundamentos y Aplicaciones Prácticas

## 1. Introducción

En el capítulo anterior exploramos modelos multimodales para el aprendizaje conjunto de imágenes y texto. Ahora, en este último capítulo, abordaremos el aprendizaje por refuerzo (RL), que representa la tercera categoría fundamental de tareas de aprendizaje automático que mencionamos al principio del libro.

El aprendizaje por refuerzo se distingue del aprendizaje supervisado y no supervisado en que se centra en aprender a través de la experiencia e interacción directa con el entorno. Este paradigma permite a los agentes desarrollar comportamientos óptimos mediante la retroalimentación recibida de sus acciones.

### 1.1 Contenido del capítulo

1. **Fundamentos de aprendizaje por refuerzo**
   * Conceptos clave y terminología
   * Formulación matemática del problema

2. **Entornos de desarrollo (Gymnasium)**
   * Instalación y configuración
   * Características principales

3. **Métodos de solución**
   * Programación dinámica en el entorno FrozenLake
   * Métodos de Monte Carlo en Blackjack
   * Aprendizaje por diferencias temporales (Q-learning)

## 2. Herramientas de Desarrollo: Gymnasium

### 2.1 Origen y evolución

[Gymnasium](https://gymnasium.farama.org/) es el sucesor oficial de OpenAI Gym, una biblioteca para el desarrollo y comparación de algoritmos de aprendizaje por refuerzo. Tras la decisión de OpenAI de discontinuar el mantenimiento de Gym, la Fundación Farama asumió el desarrollo y creó Gymnasium como su sustituto directo, preservando la compatibilidad con las versiones anteriores mientras añade mejoras significativas.

### 2.2 Características principales

Gymnasium ofrece una serie de ventajas fundamentales para el desarrollo de algoritmos de RL:

* **Interfaz unificada**: API consistente para interactuar con diversos entornos
* **Diversidad de entornos**: Desde problemas clásicos hasta simulaciones complejas
* **Facilidad de uso**: Diseñado para ser accesible para principiantes y expertos
* **Reproducibilidad**: Facilita la comparación objetiva entre algoritmos
* **Comunidad activa**: Constante evolución y mejoras

### 2.3 Instalación

Para instalar Gymnasium, podemos utilizar el gestor de paquetes pip:

```bash
# Instalación básica
pip install gymnasium

# Con entornos adicionales (recomendado para este capítulo)
pip install gymnasium[toy-text]
```

El módulo `toy-text` incluye entornos clásicos como FrozenLake y Blackjack que utilizaremos en nuestros ejemplos.

## 3. Fundamentos del Aprendizaje por Refuerzo

### 3.1 Conceptualización mediante ejemplos

El aprendizaje por refuerzo puede entenderse mediante analogías con situaciones cotidianas. Pensemos en los videojuegos clásicos como Super Mario Bros o Sonic the Hedgehog: el jugador (agente) debe navegar por niveles (entorno), tomando decisiones (acciones) como saltar o correr, basándose en lo que ve en pantalla (estados), con el objetivo de maximizar la puntuación (recompensa).

El proceso fundamental del aprendizaje por refuerzo sigue este ciclo:

1. Observación del estado actual
2. Selección y ejecución de una acción
3. Recepción de una recompensa y transición a un nuevo estado
4. Repetición hasta alcanzar un estado terminal

### 3.2 Componentes esenciales

Los elementos básicos que componen cualquier problema de aprendizaje por refuerzo son:

* **Entorno**: El mundo con el que interactúa el agente
  * *Ejemplos*: Tablero de ajedrez, simulación de carreteras, mundo virtual

* **Agente**: Entidad que toma decisiones
  * *Ejemplos*: Personaje en un videojuego, vehículo autónomo, robot

* **Estados**: Representación de la situación actual
  * *Ejemplos*: Posición en un tablero, velocidad y coordenadas de un vehículo

* **Acciones**: Decisiones disponibles para el agente
  * *Ejemplos*: Movimientos en juegos, aceleración/frenado en vehículos

* **Recompensas**: Feedback numérico que guía el aprendizaje
  * *Ejemplos*: +1 por capturar una pieza, -10 por colisionar

### 3.3 Recompensas acumuladas y factor de descuento

El objetivo del agente es maximizar la recompensa total futura (también llamada "retorno"). Sin embargo, no todas las recompensas futuras tienen el mismo valor presente.

Para modelar esta realidad, se introduce el **factor de descuento** (γ), un valor entre 0 y 1 que determina cuánto "importan" las recompensas futuras:

* γ cercano a 0: Preferencia por recompensas inmediatas
* γ cercano a 1: Valoración similar entre recompensas presentes y futuras

Matemáticamente, el retorno descontado Gt en el tiempo t se define como:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Donde:
- $G_t$ es el retorno descontado en el tiempo t
- $R_{t+k+1}$ es la recompensa recibida k+1 pasos después del tiempo t
- $\gamma$ es el factor de descuento (0 ≤ γ ≤ 1)

### 3.4 Enfoques principales

Existen dos grandes categorías de métodos para resolver problemas de aprendizaje por refuerzo:

#### 3.4.1 Enfoque basado en política

Una **política** (π) es una función que mapea estados a acciones, indicando qué acción debe tomar el agente en cada estado. Las políticas pueden ser:

* **Deterministas**: Para cada estado existe una única acción definida
* **Estocásticas**: Asignan probabilidades a diferentes acciones en cada estado

Los métodos basados en política aprenden directamente esta función de mapeo.

**Ejemplo**: Un conductor de Fórmula 1 aprende exactamente qué maniobra realizar en cada punto y condición de la pista.

#### 3.4.2 Enfoque basado en valor

Este enfoque se centra en aprender la función de valor, que estima la utilidad esperada (recompensa futura descontada) de estar en un estado particular o de realizar una acción específica en un estado determinado:

* **Función de valor de estado V(s)**: Valor esperado de estar en el estado s
* **Función de valor de acción Q(s,a)**: Valor esperado de tomar la acción a en el estado s

Las decisiones se toman eligiendo acciones que conducen a estados con mayor valor.

**Ejemplo**: Un buscador de tesoros que sigue el camino que promete la mayor recompensa, sin conocer necesariamente todas las decisiones futuras.

## 4. Resolviendo el Entorno FrozenLake con Programación Dinámica

### 4.1 Características del entorno FrozenLake

FrozenLake es un entorno discreto clásico donde un agente debe navegar por un lago congelado representado como una cuadrícula. El objetivo es moverse desde el punto de inicio hasta la meta sin caer en agujeros.

![FrozenLake Ejemplo](https://gymnasium.farama.org/_images/frozen_lake.gif)

Las características principales del entorno son:

* **Tamaño**: Disponible en versiones 4x4 y 8x8
* **Tipos de casillas**:
  - Inicio (S): Punto de partida
  - Meta (G): Objetivo, otorga recompensa +1
  - Hielo (F): Casilla segura de tránsito
  - Agujero (H): Termina el episodio sin recompensa

* **Acciones posibles**:
  - 0: Izquierda
  - 1: Abajo
  - 2: Derecha
  - 3: Arriba

* **Dinámica estocástica**: El hielo es resbaladizo, lo que significa que el agente puede "deslizarse" a una casilla diferente con cierta probabilidad.

### 4.2 Creación y simulación del entorno

Para comenzar a trabajar con FrozenLake, primero creamos una instancia del entorno:

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

# Crear el entorno
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

# Obtener dimensiones del espacio de estados y acciones
n_states = env.observation_space.n  # 16 estados para 4x4
n_actions = env.action_space.n      # 4 acciones posibles
```

Para reiniciar y visualizar el entorno:

```python
# Reiniciar el entorno
state, info = env.reset()

# Visualizar el estado actual
plt.figure(figsize=(4, 4))
plt.imshow(env.render())
plt.title(f"Estado actual: {state}")
plt.axis('off')
plt.show()
```

Ejecutar una acción y observar el resultado:

```python
# Ejecutar acción (ejemplo: mover a la derecha)
next_state, reward, terminated, truncated, info = env.step(2)

# Visualizar el nuevo estado
plt.figure(figsize=(4, 4))
plt.imshow(env.render())
plt.title(f"Nuevo estado: {next_state}, Recompensa: {reward}")
plt.axis('off')
plt.show()
```

### 4.3 Evaluación de políticas aleatorias

Antes de implementar algoritmos avanzados, evaluemos el rendimiento de políticas aleatorias:

```python
def run_episode(env, policy):
    """Ejecuta un episodio completo siguiendo una política dada."""
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Seleccionar acción según la política
        action = int(policy[state].item())
        
        # Ejecutar acción
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Acumular recompensa
        total_reward += reward
        
    return total_reward

# Generar y evaluar políticas aleatorias
n_episodes = 1000
rewards = []

for _ in range(n_episodes):
    # Crear política aleatoria (una acción por estado)
    random_policy = torch.randint(high=n_actions, size=(n_states,))
    
    # Ejecutar un episodio
    reward = run_episode(env, random_policy)
    rewards.append(reward)

# Calcular rendimiento promedio
avg_reward = sum(rewards) / n_episodes
print(f"Recompensa promedio con políticas aleatorias: {avg_reward:.4f}")
```

El resultado típico de este experimento es aproximadamente 0.016, lo que significa que solo el 1.6% de los episodios termina exitosamente con políticas aleatorias. Esto demuestra la dificultad del entorno a pesar de su aparente simplicidad.

### 4.4 Análisis de la matriz de transición

Una manera de comprender mejor el comportamiento estocástico del entorno es examinar su matriz de transición:

```python
# Examinar la matriz de transición para un estado específico
state_to_examine = 6
print("Matriz de transición para el estado 6:")
for action in range(n_actions):
    print(f"Acción {action}:")
    for trans_prob, next_state, reward, is_terminal in env.env.P[state_to_examine][action]:
        print(f"  Prob: {trans_prob}, Estado siguiente: {next_state}, " 
              f"Recompensa: {reward}, Terminal: {is_terminal}")
```

La salida muestra la probabilidad de transición a diferentes estados, dependiendo de la acción elegida. El formato es:
`(Probabilidad, Estado_siguiente, Recompensa, ¿Es_estado_terminal?)`

### 4.5 Implementación del algoritmo de iteración de valores

El algoritmo de iteración de valores es un método de programación dinámica que calcula la función de valor óptima de manera iterativa. A partir de esta función, podemos derivar la política óptima.

```python
def value_iteration(env, gamma=0.99, threshold=1e-4):
    """
    Implementación del algoritmo de iteración de valores.
    
    Args:
        env: Entorno de Gymnasium
        gamma: Factor de descuento
        threshold: Criterio de convergencia
        
    Returns:
        V: Vector de valores óptimos para cada estado
    """
    # Inicialización
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = torch.zeros(n_states)
    
    # Iteración hasta convergencia
    while True:
        # Guardar valores actuales
        V_prev = V.clone()
        
        # Actualizar valores para cada estado
        for state in range(n_states):
            # Calcular valores para cada acción
            action_values = torch.zeros(n_actions)
            
            for action in range(n_actions):
                for trans_prob, next_state, reward, _ in env.env.P[state][action]:
                    # Bellman update
                    action_values[action] += trans_prob * (reward + gamma * V_prev[next_state])
            
            # Actualizar con el máximo valor
            V[state] = torch.max(action_values)
        
        # Verificar convergencia
        max_delta = torch.max(torch.abs(V - V_prev))
        if max_delta < threshold:
            break
            
    return V
```

### 4.6 Extracción de la política óptima

Una vez que tenemos la función de valor óptima, podemos extraer la política óptima:

```python
def extract_policy(env, V, gamma=0.99):
    """
    Extrae la política óptima a partir de la función de valor.
    
    Args:
        env: Entorno de Gymnasium
        V: Vector de valores óptimos
        gamma: Factor de descuento
        
    Returns:
        policy: Vector de acciones óptimas para cada estado
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = torch.zeros(n_states)
    
    for state in range(n_states):
        # Calcular valores Q para cada acción
        Q = torch.zeros(n_actions)
        
        for action in range(n_actions):
            for trans_prob, next_state, reward, _ in env.env.P[state][action]:
                Q[action] += trans_prob * (reward + gamma * V[next_state])
        
        # Seleccionar la acción con mayor valor
        policy[state] = torch.argmax(Q)
        
    return policy
```

### 4.7 Evaluación de la política óptima

Finalmente, evaluamos el rendimiento de la política óptima:

```python
# Calcular valores y política óptima
V_optimal = value_iteration(env, gamma=0.99, threshold=1e-4)
optimal_policy = extract_policy(env, V_optimal, gamma=0.99)

# Función para evaluar la política
def evaluate_policy(env, policy, n_episodes=1000):
    """
    Evalúa el rendimiento de una política mediante simulación.
    
    Args:
        env: Entorno de Gymnasium
        policy: Vector de acciones para cada estado
        n_episodes: Número de episodios a simular
        
    Returns:
        success_rate: Tasa de éxito (porcentaje de episodios exitosos)
    """
    success_count = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = int(policy[state].item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Verificar si fue exitoso (llegó a la meta)
            if done and reward == 1.0:
                success_count += 1
                
    return success_count / n_episodes

# Evaluar la política óptima
success_rate = evaluate_policy(env, optimal_policy, n_episodes=1000)
print(f"Tasa de éxito con política óptima: {success_rate:.2%}")
```

Con la política óptima, se logra una tasa de éxito cercana al 74%, lo que representa una mejora significativa respecto al 1.6% obtenido con políticas aleatorias.

### 4.8 Implementación de iteración de políticas

Otro enfoque de programación dinámica es el algoritmo de iteración de políticas, que alterna entre evaluación y mejora de la política:

```python
def policy_evaluation(env, policy, gamma=0.99, threshold=1e-4):
    """
    Evalúa una política calculando su función de valor.
    
    Args:
        env: Entorno de Gymnasium
        policy: Vector de acciones para cada estado
        gamma: Factor de descuento
        threshold: Criterio de convergencia
        
    Returns:
        V: Vector de valores para la política dada
    """
    n_states = policy.shape[0]
    V = torch.zeros(n_states)
    
    while True:
        V_prev = V.clone()
        
        for state in range(n_states):
            action = int(policy[state].item())
            
            # Calcular valor esperado para la acción dada
            value = 0
            for trans_prob, next_state, reward, _ in env.env.P[state][action]:
                value += trans_prob * (reward + gamma * V_prev[next_state])
                
            V[state] = value
            
        # Verificar convergencia
        if torch.max(torch.abs(V - V_prev)) < threshold:
            break
            
    return V

def policy_improvement(env, V, gamma=0.99):
    """
    Mejora la política basándose en la función de valor actual.
    
    Args:
        env: Entorno de Gymnasium
        V: Vector de valores para cada estado
        gamma: Factor de descuento
        
    Returns:
        policy: Vector de acciones mejoradas para cada estado
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = torch.zeros(n_states)
    
    for state in range(n_states):
        Q = torch.zeros(n_actions)
        
        for action in range(n_actions):
            for trans_prob, next_state, reward, _ in env.env.P[state][action]:
                Q[action] += trans_prob * (reward + gamma * V[next_state])
                
        policy[state] = torch.argmax(Q)
        
    return policy

def policy_iteration(env, gamma=0.99, threshold=1e-4):
    """
    Implementación del algoritmo de iteración de políticas.
    
    Args:
        env: Entorno de Gymnasium
        gamma: Factor de descuento
        threshold: Criterio de convergencia
        
    Returns:
        V: Vector de valores óptimos
        policy: Vector de acciones óptimas
    """
    # Inicialización con política aleatoria
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = torch.randint(high=n_actions, size=(n_states,)).float()
    
    while True:
        # Evaluación de política
        V = policy_evaluation(env, policy, gamma, threshold)
        
        # Mejora de política
        new_policy = policy_improvement(env, V, gamma)
        
        # Verificar convergencia
        if torch.equal(new_policy, policy):
            return V, new_policy
            
        policy = new_policy
```

## 5. Aprendizaje Monte Carlo con el Entorno Blackjack

### 5.1 Métodos de aprendizaje sin modelo

A diferencia de los métodos de programación dinámica (value iteration, policy iteration), los métodos de aprendizaje por refuerzo sin modelo (model-free) no requieren conocimiento explícito de las probabilidades de transición ni de las recompensas del entorno. Estos métodos aprenden directamente a través de la experiencia, recopilando información al interactuar con el entorno.

Monte Carlo (MC) es uno de estos métodos, caracterizado por:

* Aprender exclusivamente de episodios completos de experiencia
* No requerir conocimiento previo del modelo del entorno
* Actualizar estimaciones basándose en retornos reales observados
* Ser especialmente eficaz en entornos con información parcial o estocásticos

### 5.2 Descripción del entorno Blackjack

Blackjack es un juego de cartas donde el jugador compite contra el crupier. El objetivo es obtener una mano cuyo valor se acerque lo más posible a 21 sin pasarse.

![Blackjack](https://gymnasium.farama.org/_images/blackjack.gif)

Características principales del entorno en Gymnasium:

* **Estado**: Tripla (suma_jugador, carta_visible_crupier, as_usable)
  - suma_jugador: Valor total de las cartas del jugador
  - carta_visible_crupier: Valor de la carta visible del crupier
  - as_usable: Booleano que indica si el jugador tiene un As que cuenta como 11

* **Acciones**:
  - 0: Plantarse (stick)
  - 1: Pedir carta (hit)

* **Recompensas**:
  - +1: Victoria del jugador
  - 0: Empate
  - -1: Victoria del crupier

* **Reglas de cartas**:
  - Cartas numéricas (2-10): Valor nominal
  - Figuras (J, Q, K): 10 puntos
  - As: 1 u 11 puntos (11 si no causa que se supere 21)

### 5.3 Creación y simulación del entorno

```python
# Crear entorno Blackjack
env = gym.make('Blackjack-v1')

# Reiniciar entorno
state, _ = env.reset(seed=42)
print(f"Estado inicial: {state}")
# Ejemplo: (14, 10, False) - Jugador tiene 14, crupier muestra 10, sin As usable

# Simular algunas acciones
print("\nSimulación de un episodio:")
done = False
total_reward = 0

while not done:
    # Pedir carta (hit)
    state, reward, terminated, truncated, _ = env.step(1)
    print(f"Acción: Pedir carta -> Estado: {state}, Recompensa: {reward}")
    
    # Si tenemos 18 o más, nos plantamos
    if state[0] >= 18 or terminated or truncated:
        if not (terminated or truncated):
            state, reward, terminated, truncated, _ = env.step(0)
            print(f"Acción: Plantarse -> Estado: {state}, Recompensa: {reward}")
        
        done = terminated or truncated
        total_reward = reward

print(f"\nRecompensa final: {total_reward}")
```

### 5.4 Evaluación de políticas con Monte Carlo First-Visit

El algoritmo Monte Carlo first-visit evalúa una política estimando la función de valor de estado mediante la recopilación de retornos promedio de la primera vez que se visita cada estado en múltiples episodios.

```python
def run_blackjack_episode(env, policy):
    """
    Ejecuta un episodio completo de Blackjack siguiendo una política fija.
    
    Args:
        env: Entorno Blackjack
        policy: Función que toma un estado y devuelve una acción (0=stick, 1=hit)
        
    Returns:
        states: Lista de estados visitados
        rewards: Lista de recompensas recibidas
    """
    states = []
    rewards = []
    
    state, _ = env.reset()
    done = False
    
    while not done:
        states.append(state)
        
        # Determinar acción según la política
        action = policy(state)
        
        # Ejecutar acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        state = next_state
        
    return states, rewards

def mc_prediction_first_visit(env, policy, gamma=1.0, n_episodes=500000):
    """
    Estimación Monte Carlo first-visit de la función de valor de estado.
    
    Args:
        env: Entorno Blackjack
        policy: Función que toma un estado y devuelve una acción
        gamma: Factor de descuento
        n_episodes: Número de episodios a simular
        
    Returns:
        V: Diccionario con los valores estimados para cada estado
    """
    # Inicialización
    returns_sum = {}  # Suma de retornos para cada estado
    returns_count = {}  # Contador de visitas para cada estado
    V = {}  # Valor estimado para cada estado
    
    for i in range(n_episodes):
        # Mostrar progreso
        if i % 100000 == 0:
            print(f"Episodio {i}/{n_episodes}")
            
        # Ejecutar un episodio
        states, rewards = run_blackjack_episode(env, policy)
        
        # Calcular retornos para cada paso
        G = 0
        states.reverse()  # Invertir para procesar desde el final
        rewards.reverse()
        
        # Procesar cada paso
        for t, (state, reward) in enumerate(zip(states, rewards)):
            G = gamma * G + reward
            
            # First-visit: solo consideramos la primera vez que vemos el estado
            if state not in states[:t]:
                # Actualizar contadores
                if state not in returns_sum:
                    returns_sum[state] = 0
                    returns_count[state] = 0
                    
                returns_sum[state] += G
                returns_count[state] += 1
                
                # Actualizar estimación del valor
                V[state] = returns_sum[state] / returns_count[state]
                
    return V
```

### 5.5 Evaluación de una política simple

Vamos a evaluar una política simple: pedir carta (hit) hasta alcanzar al menos 18 puntos, luego plantarse (stick).

```python
# Definir política simple
def simple_policy(state):
    player_sum, dealer_card, usable_ace = state
    return 1 if player_sum < 18 else 0  # hit si < 18, stick en caso contrario

# Evaluar la política
V = mc_prediction_first_visit(env, simple_policy, gamma=1.0, n_episodes=500000)

# Verificar número de estados evaluados
print(f"Número de estados evaluados: {len(V)}")

# Imprimir algunos valores de ejemplo
print("\nEjemplos de valores de estado:")
examples = [(13, 10, False), (19, 7, True), (18, 7, False)]
for state in examples:
    if state in V:
        print(f"Estado {state}: {V[state]:.3f}")
```

### 5.6 Visualización de la función de valor

Podemos visualizar la función de valor para entender mejor la política:

```python
import numpy as np

# Preparar matrices para visualización
player_sums = np.arange(12, 22)
dealer_cards = np.arange(1, 11)

# Matriz para As usable
value_usable = np.zeros((len(player_sums), len(dealer_cards)))
for i, player_sum in enumerate(player_sums):
    for j, dealer_card in enumerate(dealer_cards):
        state = (player_sum, dealer_card, True)
        if state in V:
            value_usable[i, j] = V[state]
        else:
            value_usable[i, j] = 0
            
# Matriz para As no usable
value_no_usable = np.zeros((len(player_sums), len(dealer_cards)))
for i, player_sum in enumerate(player_sums):
    for j, dealer_card in enumerate(dealer_cards):
        state = (player_sum, dealer_card, False)
        if state in V:
            value_no_usable[i, j] = V[state]
        else:
            value_no_usable[i, j] = 0

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Con As usable
im0 = axes[0].imshow(value_usable, cmap='viridis', extent=[1, 10, 12, 21])
axes[0].set_xlabel('Carta visible del crupier')
axes[0].set_ylabel('Suma del jugador')
axes[0].set_title('Valor con As usable')
fig.colorbar(im0, ax=axes[0])

# Sin As usable
im1 = axes[1].imshow(value_no_usable, cmap='viridis', extent=[1, 10, 12, 21])
axes[1].set_xlabel('Carta visible del crupier')
axes[1].set_ylabel('Suma del jugador')
axes[1].set_title('Valor sin As usable')
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
```

## 6. Control Monte Carlo On-Policy para Blackjack

### 6.1 De la evaluación al control

Si bien la evaluación de políticas nos ayuda a entender qué tan buena es una política fija, el objetivo final del aprendizaje por refuerzo es encontrar la política óptima. El **control Monte Carlo on-policy** extiende el enfoque de evaluación para mejorar iterativamente la política mientras la evaluamos.

El proceso sigue un ciclo de:
1. Evaluación de la política actual
2. Mejora de la política basada en los valores aprendidos
3. Repetición hasta convergencia

A diferencia de los métodos de programación dinámica, el control Monte Carlo on-policy aprende directamente de la experiencia sin requerir un modelo del entorno.

### 6.2 Algoritmo de control Monte Carlo on-policy

```python
def mc_control_on_policy(env, gamma=1.0, n_episodes=500000):
    """
    Control Monte Carlo on-policy para encontrar una política óptima.
    
    Args:
        env: Entorno Blackjack
        gamma: Factor de descuento
        n_episodes: Número de episodios a simular
        
    Returns:
        Q: Diccionario con valores Q para cada par estado-acción
        policy: Política óptima resultante
    """
    # Inicialización
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    # Política epsilon-greedy
    def epsilon_greedy_policy(state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(env.action_space.n)  # Exploración
        else:
            return np.argmax(Q[state])  # Explotación
    
    for i in range(n_episodes):
        # Mostrar progreso
        if i % 100000 == 0:
            print(f"Episodio {i}/{n_episodes}")
        
        # Generar episodio
        episode = []
        state, _ = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            state = next_state
        
        # Extraer estados y acciones únicas del episodio
        state_action_pairs = set([(s, a) for s, a, _ in episode])
        
        # Actualizar valores Q para cada par (estado, acción) del episodio
        for state, action, _ in episode:
            # Solo considerar pares válidos (jugador no ha superado 21)
            if state[0] <= 21 and (state, action) in state_action_pairs:
                # First-visit: extraer el retorno desde este paso
                first_occurrence_idx = next(i for i, x in enumerate(episode) 
                                         if x[0] == state and x[1] == action)
                
                # Calcular retorno desde primera ocurrencia
                G = sum([gamma**i * r for i, (_, _, r) 
                       in enumerate(episode[first_occurrence_idx:])])
                
                # Actualizar estimaciones
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
                
                # Eliminar par para no contarlo nuevamente (first-visit)
                state_action_pairs.remove((state, action))
    
    # Extraer política óptima final
    policy = {}
    for state in Q.keys():
        policy[state] = np.argmax(Q[state])
        
    return Q, policy
```

### 6.3 Entrenamiento y evaluación

```python
# Entrenar el agente
Q_optimal, optimal_policy = mc_control_on_policy(env, gamma=1.0, n_episodes=500000)

# Definir función de simulación para cualquier política
def simulate_episode(env, policy):
    """
    Simula un episodio completo siguiendo una política dada.
    
    Args:
        env: Entorno Blackjack
        policy: Diccionario que mapea estados a acciones
        
    Returns:
        reward: Recompensa final del episodio
    """
    state, _ = env.reset()
    done = False
    
    while not done:
        # La política puede no tener todas las combinaciones posibles,
        # en ese caso usamos una acción por defecto (0 = stick)
        action = policy.get(state, 0)
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if done:
            return reward

# Comparar política óptima vs política simple
n_episodes = 100000
rewards_optimal = []
rewards_simple = []

print("Evaluando políticas...")
for _ in range(n_episodes):
    # Evaluar política óptima
    rewards_optimal.append(simulate_episode(env, optimal_policy))
    
    # Evaluar política simple (plantarse en 18)
    state, _ = env.reset()
    done = False
    while not done:
        action = 1 if state[0] < 18 else 0
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            rewards_simple.append(reward)

# Calcular tasas de victoria
win_rate_optimal = sum(r == 1 for r in rewards_optimal) / n_episodes
win_rate_simple = sum(r == 1 for r in rewards_simple) / n_episodes

print(f"Tasa de victoria con política simple: {win_rate_simple:.4f}")
print(f"Tasa de victoria con política óptima: {win_rate_optimal:.4f}")
print(f"Mejora: {(win_rate_optimal - win_rate_simple) * 100:.2f}%")
```

### 6.4 Visualización de la política óptima

Para entender mejor la política óptima, podemos visualizarla:

```python
# Preparar matrices para visualización
player_sums = np.arange(12, 22)
dealer_cards = np.arange(1, 11)

# Matriz para As usable
policy_usable = np.zeros((len(player_sums), len(dealer_cards)))
for i, player_sum in enumerate(player_sums):
    for j, dealer_card in enumerate(dealer_cards):
        state = (player_sum, dealer_card, True)
        if state in optimal_policy:
            policy_usable[i, j] = optimal_policy[state]
        else:
            policy_usable[i, j] = 0
            
# Matriz para As no usable
policy_no_usable = np.zeros((len(player_sums), len(dealer_cards)))
for i, player_sum in enumerate(player_sums):
    for j, dealer_card in enumerate(dealer_cards):
        state = (player_sum, dealer_card, False)
        if state in optimal_policy:
            policy_no_usable[i, j] = optimal_policy[state]
        else:
            policy_no_usable[i, j] = 0

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Con As usable
im0 = axes[0].imshow(policy_usable, cmap='Accent', extent=[1, 10, 12, 21])
axes[0].set_xlabel('Carta visible del crupier')
axes[0].set_ylabel('Suma del jugador')
axes[0].set_title('Política óptima con As usable')
axes[0].set_xticks(np.arange(1, 11))
axes[0].set_yticks(np.arange(12, 22))
cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0, 1])
cbar0.ax.set_yticklabels(['Plantarse', 'Pedir carta'])

# Sin As usable
im1 = axes[1].imshow(policy_no_usable, cmap='Accent', extent=[1, 10, 12, 21])
axes[1].set_xlabel('Carta visible del crupier')
axes[1].set_ylabel('Suma del jugador')
axes[1].set_title('Política óptima sin As usable')
axes[1].set_xticks(np.arange(1, 11))
axes[1].set_yticks(np.arange(12, 22))
cbar1 = fig.colorbar(im1, ax=axes[1], ticks=[0, 1])
cbar1.ax.set_yticklabels(['Plantarse', 'Pedir carta'])

plt.tight_layout()
plt.show()
```

Esta visualización revela patrones interesantes en la estrategia óptima:
- Con sumas bajas (12-16), generalmente conviene pedir carta
- Con sumas altas (19-21), lo mejor es plantarse
- Las decisiones para sumas intermedias (17-18) dependen de la carta visible del crupier
- La estrategia varía significativamente cuando se tiene un As usable

## 7. Aprendizaje por Diferencias Temporales: Q-learning

### 7.1 Limitaciones de Monte Carlo y ventajas de TD

El método de Monte Carlo tiene dos limitaciones principales:
1. Requiere esperar hasta el final de cada episodio para actualizar los valores
2. Puede tener alta varianza en las estimaciones debido a la naturaleza aleatoria de los episodios completos

El aprendizaje por diferencias temporales (TD Learning) supera estas limitaciones al:
- Actualizar estimaciones después de cada paso (sin esperar al final del episodio)
- Utilizar estimaciones existentes para reducir la varianza (bootstrapping)
- Converger generalmente más rápido que Monte Carlo en muchos problemas

### 7.2 Q-learning: un algoritmo TD off-policy

Q-learning es un algoritmo TD que aprende la función de valor óptima Q* directamente, independientemente de la política que se esté siguiendo (off-policy). Esto significa que puede aprender la política óptima mientras explora el entorno con cualquier estrategia de exploración.

La actualización clave en Q-learning se basa en la ecuación de Bellman:

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a)]$$

Donde:
- $\alpha$ es la tasa de aprendizaje
- $\gamma$ es el factor de descuento
- $r$ es la recompensa inmediata
- $s'$ es el siguiente estado
- $\max_{a'} Q(s',a')$ es el valor máximo posible en el siguiente estado

### 7.3 Implementación de Q-learning para Blackjack

```python
def q_learning(env, gamma=1.0, alpha=0.01, epsilon=1.0, 
               final_epsilon=0.1, n_episodes=10000):
    """
    Implementación de Q-learning para el entorno Blackjack.
    
    Args:
        env: Entorno Blackjack
        gamma: Factor de descuento
        alpha: Tasa de aprendizaje
        epsilon: Probabilidad inicial de exploración
        final_epsilon: Probabilidad final de exploración
        n_episodes: Número de episodios a simular
        
    Returns:
        Q: Diccionario con valores Q para cada par estado-acción
        policy: Política óptima resultante
        rewards: Lista de recompensas por episodio
    """
    # Inicialización
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards_per_episode = np.zeros(n_episodes)
    
    # Decaimiento de epsilon
    epsilon_decay = (epsilon - final_epsilon) / (n_episodes / 2)
    
    for i in range(n_episodes):
        # Actualizar epsilon (decaimiento linear)
        if epsilon > final_epsilon:
            epsilon -= epsilon_decay
            
        # Iniciar episodio
        state, _ = env.reset()
        done = False
        
        while not done:
            # Política epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.choice(env.action_space.n)  # Exploración
            else:
                action = np.argmax(Q[state])  # Explotación
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualización Q-learning
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            # Transición al siguiente estado
            state = next_state
            
            # Registrar recompensa
            if done:
                rewards_per_episode[i] = reward
        
        # Mostrar progreso
        if (i+1) % 1000 == 0:
            print(f"Episodio {i+1}/{n_episodes}, Epsilon: {epsilon:.4f}")
    
    # Extraer política óptima
    policy = {}
    for state in Q.keys():
        policy[state] = np.argmax(Q[state])
        
    return Q, policy, rewards_per_episode
```

### 7.4 Entrenamiento y evaluación del agente Q-learning

```python
# Entrenar el agente
Q_values, q_policy, rewards = q_learning(env, 
                                         gamma=1.0, 
                                         alpha=0.01, 
                                         epsilon=1.0,
                                         final_epsilon=0.1,
                                         n_episodes=100000)

# Calcular media móvil para visualizar el progreso del aprendizaje
def moving_average(data, window_size=100):
    """Calcula la media móvil de los datos."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Visualizar curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(moving_average(rewards, 1000))
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('Episodio')
plt.ylabel('Recompensa promedio (media móvil)')
plt.title('Curva de aprendizaje de Q-learning')
plt.ylim(-1.1, 1.1)
plt.grid(alpha=0.3)
plt.show()

# Evaluar la política aprendida
n_test_episodes = 100000
test_rewards = []

for _ in range(n_test_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = q_policy.get(state, 0)  # Acción por defecto: plantarse
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if done:
            test_rewards.append(reward)

# Calcular estadísticas
win_rate = sum(r == 1 for r in test_rewards) / n_test_episodes
loss_rate = sum(r == -1 for r in test_rewards) / n_test_episodes
draw_rate = sum(r == 0 for r in test_rewards) / n_test_episodes

print(f"Resultados tras {n_test_episodes} episodios:")
print(f"  • Victorias: {win_rate:.4f}")
print(f"  • Derrotas: {loss_rate:.4f}")
print(f"  • Empates: {draw_rate:.4f}")
```

### 7.5 Comparación entre los métodos estudiados

Para completar nuestro análisis, comparemos los tres enfoques estudiados:

```python
# Simulamos episodios con las tres políticas
n_eval_episodes = 100000
results = {
    'Política simple (plantarse en 18)': [],
    'Política Monte Carlo on-policy': [],
    'Política Q-learning': []
}

print("Evaluando las tres políticas...")
for _ in range(n_eval_episodes):
    # Reiniciar entorno para cada episodio
    env.reset()
    
    # 1. Política simple
    state, _ = env.reset()
    done = False
    while not done:
        action = 1 if state[0] < 18 else 0
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            results['Política simple (plantarse en 18)'].append(reward)
    
    # 2. Política Monte Carlo
    state, _ = env.reset()
    done = False
    while not done:
        action = optimal_policy.get(state, 0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            results['Política Monte Carlo on-policy'].append(reward)
    
    # 3. Política Q-learning
    state, _ = env.reset()
    done = False
    while not done:
        action = q_policy.get(state, 0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            results['Política Q-learning'].append(reward)    # Calcular tasas de victoria
performance = {}
for method, rewards in results.items():
    win_rate = sum(r == 1 for r in rewards) / n_eval_episodes
    performance[method] = win_rate

# Visualizar comparación
plt.figure(figsize=(12, 6))
methods = list(performance.keys())
win_rates = [performance[method] for method in methods]

plt.bar(methods, win_rates, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('Tasa de victoria')
plt.title('Comparación de rendimiento entre métodos')
plt.ylim(0, 0.5)

for i, v in enumerate(win_rates):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

plt.tight_layout()
plt.show()
```

### 7.6 Evaluación de la política de Q-learning

Para evaluar cuantitativamente el rendimiento de la política aprendida con Q-learning, simulamos múltiples episodios:

```python
n_episode = 100000
n_win_opt = 0

for _ in range(n_episode):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_opt += 1

print(f'Probabilidad de ganar con Q-learning: {n_win_opt / n_episode:.5f}')
```

Este experimento generalmente muestra una tasa de victoria cercana al 42.4%, lo que representa una mejora significativa respecto a la política simple de plantarse en 18.

### 7.7 Conclusiones sobre Q-learning

El algoritmo Q-learning ofrece varias ventajas importantes para problemas de aprendizaje por refuerzo:

* **Aprendizaje incremental**: Actualiza estimaciones después de cada paso sin esperar al final del episodio
* **Independencia del modelo**: No requiere conocimiento previo del entorno
* **Exploración eficiente**: La política epsilon-greedy permite balancear exploración y explotación
* **Convergencia**: Bajo ciertas condiciones, converge a la política óptima

Cuando comparamos los tres enfoques estudiados (política simple, Monte Carlo y Q-learning), observamos que tanto Monte Carlo como Q-learning superan significativamente a la política simple, mientras mantienen un rendimiento similar entre ellos para el problema de Blackjack.

## 8. Síntesis y aplicación práctica

### 8.1 Conclusión general

En este capítulo hemos explorado diferentes enfoques del aprendizaje por refuerzo:

* Comenzamos con los fundamentos teóricos: entornos, agentes, estados, acciones y recompensas
* Implementamos métodos de programación dinámica que requieren conocer completamente el modelo del entorno (FrozenLake)
* Avanzamos hacia métodos sin modelo como Monte Carlo, que aprenden directamente de episodios completos (Blackjack)
* Finalmente, estudiamos Q-learning, que combina las ventajas de la actualización inmediata con la independencia del modelo

Cada enfoque tiene sus fortalezas y limitaciones, siendo apropiado para diferentes tipos de problemas. La programación dinámica es ideal cuando conocemos perfectamente el entorno, Monte Carlo funciona bien en entornos estocásticos con episodios bien definidos, y Q-learning destaca en situaciones donde queremos aprendizaje inmediato sin depender del conocimiento del modelo.

El aprendizaje por refuerzo continúa siendo un campo en rápida evolución, con aplicaciones que van desde videojuegos hasta robótica, finanzas y sistemas de recomendación, proporcionando un marco poderoso para crear agentes inteligentes capaces de tomar decisiones óptimas en entornos complejos.

### 8.2 Ejercicios propuestos

#### 8.2.1 Resolución del entorno FrozenLake 8x8
El primer ejercicio propuesto consiste en resolver el entorno FrozenLake en su versión ampliada de 8x8, utilizando los algoritmos de iteración de valores o iteración de políticas. Esta implementación nos permitirá verificar la escalabilidad de los métodos de programación dinámica en entornos más complejos.

```python
import gymnasium as gym
import torch
import numpy as np

# Inicializar entorno
env = gym.make("FrozenLake8x8-v1", is_slippery=True)

def policy_evaluation(env, policy, gamma=0.99, threshold=1e-4):
    """Evalúa una política dada calculando su función de valor."""
    n_states = policy.shape[0]
    V = torch.zeros(n_states)
    
    while True:
        V_prev = V.clone()
        
        for state in range(n_states):
            action = int(policy[state].item())
            value = 0
            
            for trans_prob, next_state, reward, _ in env.env.P[state][action]:
                value += trans_prob * (reward + gamma * V_prev[next_state])
                
            V[state] = value
            
        # Verificar convergencia
        if torch.max(torch.abs(V - V_prev)) < threshold:
            break
            
    return V

def policy_improvement(env, V, gamma=0.99):
    """Mejora la política basándose en la función de valor actual."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = torch.zeros(n_states)
    
    for state in range(n_states):
        Q = torch.zeros(n_actions)
        
        for action in range(n_actions):
            for trans_prob, next_state, reward, _ in env.env.P[state][action]:
                Q[action] += trans_prob * (reward + gamma * V[next_state])
                
        policy[state] = torch.argmax(Q)
        
    return policy

def policy_iteration(env, gamma=0.99, threshold=1e-4):
    """Implementación completa del algoritmo de iteración de políticas."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Inicializar con política aleatoria
    policy = torch.randint(high=n_actions, size=(n_states,)).float()
    
    iteration = 0
    while True:
        iteration += 1
        
        # Paso 1: Evaluación de política
        V = policy_evaluation(env, policy, gamma, threshold)
        
        # Paso 2: Mejora de política
        new_policy = policy_improvement(env, V, gamma)
        
        # Paso 3: Verificar convergencia
        if torch.equal(new_policy, policy):
            print(f"Política convergió después de {iteration} iteraciones")
            return V, new_policy
            
        policy = new_policy

# Obtener política óptima
_, optimal_policy = policy_iteration(env)

# Función para evaluar el rendimiento
def evaluate_policy(env, policy, n_episodes=1000):
    """Evalúa una política mediante simulación de episodios."""
    success_count = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = int(policy[state].item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done and reward == 1.0:
                success_count += 1
                
    return success_count / n_episodes

# Evaluar rendimiento
success_rate = evaluate_policy(env, optimal_policy, n_episodes=1000)
print(f"✓ Tasa de éxito con política óptima en FrozenLake8x8: {success_rate:.2%}")
```
```

---

### ✅ 1B. Policy Iteration en `FrozenLake8x8-v1`

```python
def policy_evaluation(env, policy, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = int(policy[state].item())
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob * (reward + gamma * V[new_state])
        if torch.max(torch.abs(V - V_temp)) < threshold:
            break
        V = V_temp.clone()
    return V

def policy_improvement(env, V, gamma):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        q = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                q[action] += trans_prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(q)
    return policy

def policy_iteration(env, gamma, threshold):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        new_policy = policy_improvement(env, V, gamma)
        if torch.equal(new_policy, policy):
            return V, new_policy
        policy = new_policy

# Ejecutar
V_pi, policy_pi = policy_iteration(env, gamma, threshold)
print("Ejercicio 1B - Policy Iteration completado.")
```

#### 8.2.2 Implementación de Monte Carlo Every-Visit

El segundo ejercicio propuesto consiste en implementar la variante Every-Visit del método de evaluación Monte Carlo, que a diferencia del método First-Visit, actualiza el valor de un estado cada vez que aparece en un episodio, no solo la primera vez. Esta implementación puede proporcionar estimaciones más precisas en ciertos contextos.

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Crear entorno Blackjack
env = gym.make("Blackjack-v1")

def run_episode(env, policy):
    """
    Ejecuta un episodio completo siguiendo una política determinada.
    
    Args:
        env: Entorno Blackjack
        policy: Función que toma un estado y devuelve una acción
        
    Returns:
        states: Lista de estados visitados
        rewards: Lista de recompensas recibidas
    """
    state, _ = env.reset()
    states = [state]
    rewards = []
    done = False
    
    while not done:
        # Determinar acción según la política
        action = policy(state)
        
        # Ejecutar acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Registrar estado y recompensa
        states.append(next_state)
        rewards.append(reward)
        
        # Actualizar estado
        state = next_state
        
    return states, rewards

def mc_every_visit(env, policy, gamma=1.0, n_episodes=500000):
    """
    Evaluación Monte Carlo every-visit de la función de valor.
    
    Args:
        env: Entorno Blackjack
        policy: Función que toma un estado y devuelve una acción
        gamma: Factor de descuento
        n_episodes: Número de episodios a simular
        
    Returns:
        V: Diccionario con los valores estimados para cada estado
    """
    # Inicialización
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    for i in range(n_episodes):
        # Mostrar progreso
        if i % 100000 == 0:
            print(f"Episodio {i}/{n_episodes}")
        
        # Ejecutar episodio
        states, rewards = run_episode(env, policy)
        
        # Calcular retornos
        G = 0
        for t in range(len(rewards) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            
            # Every-visit: actualizar cada vez que visitamos el estado
            if state[0] <= 21:  # Solo estados válidos (jugador no ha perdido)
                returns_sum[state] += G
                returns_count[state] += 1
    
    # Calcular valores promedio
    V = {}
    for state in returns_sum:
        V[state] = returns_sum[state] / returns_count[state]
        
    return V

# Definir política simple: Pedir carta hasta sumar 18 o más
def simple_policy(state):
    player_sum, dealer_card, usable_ace = state
    return 1 if player_sum < 18 else 0  # 1=hit, 0=stick

# Ejecutar el algoritmo
V = mc_every_visit(env, simple_policy, gamma=1.0, n_episodes=500000)

# Visualizar resultados
def visualize_value_function(V):
    # Preparar matrices para visualización
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    
    # Separar por uso de As
    V_usable = np.zeros((len(player_sums), len(dealer_cards)))
    V_no_usable = np.zeros((len(player_sums), len(dealer_cards)))
    
    # Llenar matrices
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            # Con As usable
            if (player_sum, dealer_card, True) in V:
                V_usable[i, j] = V[(player_sum, dealer_card, True)]
            
            # Sin As usable
            if (player_sum, dealer_card, False) in V:
                V_no_usable[i, j] = V[(player_sum, dealer_card, False)]
    
    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Con As usable
    im0 = axes[0].imshow(V_usable, cmap='viridis', extent=[1, 10, 12, 21])
    axes[0].set_title('Función de valor con As usable')
    axes[0].set_xlabel('Carta visible del crupier')
    axes[0].set_ylabel('Suma del jugador')
    fig.colorbar(im0, ax=axes[0])
    
    # Sin As usable
    im1 = axes[1].imshow(V_no_usable, cmap='viridis', extent=[1, 10, 12, 21])
    axes[1].set_title('Función de valor sin As usable')
    axes[1].set_xlabel('Carta visible del crupier')
    axes[1].set_ylabel('Suma del jugador')
    fig.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

# Mostrar resultados
print(f"Número de estados evaluados: {len(V)}")
visualize_value_function(V)
```
```

## 9. Resumen y Conclusiones

### 9.1 Conceptos y métodos fundamentales del aprendizaje por refuerzo

El aprendizaje por refuerzo representa un paradigma único dentro del aprendizaje automático, centrado en la toma de decisiones secuenciales a través de la interacción directa con el entorno. A lo largo de este capítulo hemos explorado:

1. **Fundamentos teóricos**:
   - El ciclo de interacción agente-entorno
   - Formulación matemática de los procesos de decisión de Markov (MDP)
   - Funciones de valor y políticas

2. **Métodos de programación dinámica**:
   - Iteración de valores
   - Iteración de políticas

3. **Métodos libres de modelo**:
   - Monte Carlo (first-visit y every-visit)
   - Aprendizaje por diferencias temporales (Q-learning)

### 9.2 Tabla comparativa de algoritmos

| Algoritmo | Tipo | Requiere modelo | Eficiencia muestral | Actualización | Convergencia | Aplicabilidad |
|-----------|------|-----------------|---------------------|---------------|--------------|---------------|
| Value Iteration | Prog. dinámica | Sí | Alta | Por iteración | Garantizada | Entornos pequeños con modelo conocido |
| Policy Iteration | Prog. dinámica | Sí | Alta | Por iteración | Garantizada | Entornos pequeños con modelo conocido |
| Monte Carlo | Model-free | No | Baja | Al final del episodio | Asintótica | Problemas episódicos |
| Q-learning | Model-free | No | Media | Por paso | Asintótica | Amplio rango de problemas |

### 9.3 Resultados comparativos

En nuestras implementaciones prácticas, hemos observado los siguientes resultados:

- **FrozenLake-v1 (4x4)**:
  - Política aleatoria: ~1.6% de tasa de éxito
  - Política óptima (programación dinámica): ~74% de tasa de éxito

- **FrozenLake8x8-v1**:
  - Política óptima (programación dinámica): ~75% de tasa de éxito

- **Blackjack-v1**:
  - Política simple (plantarse en 18): ~40% de victorias
  - Política Monte Carlo: ~43% de victorias
  - Política Q-learning: ~42.4% de victorias

### 9.4 Aplicaciones y perspectivas futuras

El aprendizaje por refuerzo ha demostrado ser extraordinariamente potente en una amplia gama de aplicaciones:

- **Robótica**: Control motor y manipulación de objetos
- **Videojuegos**: Agentes que superan el rendimiento humano (AlphaGo, AlphaStar)
- **Optimización de sistemas**: Gestión de recursos, redes eléctricas
- **Finanzas**: Trading algorítmico y gestión de carteras
- **Medicina**: Dosificación personalizada y planes de tratamiento

Las tendencias actuales apuntan hacia métodos más escalables, que integran aprendizaje profundo con RL (Deep Reinforcement Learning), y algoritmos más eficientes en términos de muestras.

### 9.5 Recursos adicionales para profundizar

Para lectores interesados en expandir su conocimiento, recomendamos:

1. **Libros**:
   - "Reinforcement Learning: An Introduction" (Sutton & Barto)
   - "Deep Reinforcement Learning Hands-On" (Maxim Lapan)

2. **Cursos en línea**:
   - CS234: Reinforcement Learning (Stanford)
   - Deep RL Bootcamp (Berkeley)

3. **Frameworks**:
   - Gymnasium (sucesor de OpenAI Gym)
   - Stable Baselines3
   - TensorFlow Agents
   - RLlib

El aprendizaje por refuerzo sigue siendo un campo en rápida evolución, con nuevos algoritmos y aplicaciones emergiendo constantemente. Las bases que hemos explorado en este capítulo proporcionan un sólido fundamento para comprender y contribuir a estos avances.

---

*Este capítulo fue elaborado como parte del material educativo avanzado sobre inteligencia artificial y aprendizaje automático.*

---

## ✅ Código para evaluar una política en FrozenLake8x8

Este código usa **policy iteration** para obtener la política óptima y luego la **evalúa simulando 1,000 episodios**.

```python
import gymnasium as gym
import torch
import numpy as np

# Inicializar entorno
env = gym.make("FrozenLake8x8-v1", is_slippery=True)

# Funciones de policy iteration
def policy_evaluation(env, policy, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = int(policy[state].item())
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob * (reward + gamma * V[new_state])
        if torch.max(torch.abs(V - V_temp)) < threshold:
            break
        V = V_temp.clone()
    return V

def policy_improvement(env, V, gamma):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        q = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                q[action] += trans_prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(q)
    return policy

def policy_iteration(env, gamma=0.99, threshold=1e-4):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        new_policy = policy_improvement(env, V, gamma)
        if torch.equal(new_policy, policy):
            return V, new_policy
        policy = new_policy

# Obtener política óptima
_, optimal_policy = policy_iteration(env)

# Simular política óptima
def evaluate_policy(env, policy, n_episodes=1000):
    success_count = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(policy[state].item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done and reward == 1.0:
                success_count += 1
    return success_count / n_episodes

# Evaluar rendimiento
success_rate = evaluate_policy(env, optimal_policy, n_episodes=1000)
print(f"✔️ Tasa de éxito con política óptima en FrozenLake8x8: {success_rate:.2%}")
```

---

Este código debería darte un resultado como:

```
✔️ Tasa de éxito con política óptima en FrozenLake8x8: 75.3%
```
