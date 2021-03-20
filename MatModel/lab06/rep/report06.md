---
# Front matter
lang: ru-RU
title: "Отчет лабораторной № 6"
subtitle: "Задача об эпидемии"
author: "Хохлачева Яна Дмитриевна"

# Formatting
toc-title: "Содержание"
toc: true # Table of contents
toc_depth: 2
lof: true # List of figures
lot: true # List of tables
fontsize: 12pt
linestretch: 1.5
papersize: a4paper
documentclass: scrreprt
polyglossia-lang: russian
polyglossia-otherlangs: english
mainfont: PT Serif
romanfont: PT Serif
sansfont: PT Sans
monofont: PT Mono
mainfontoptions: Ligatures=TeX
romanfontoptions: Ligatures=TeX
indent: true
pdf-engine: lualatex
header-includes:
  - \linepenalty=10 # the penalty added to the badness of each line within a paragraph (no associated penalty node) Increasing the value makes tex try to have fewer lines in the paragraph.
  - \interlinepenalty=0 # value of the penalty (node) added after each line of a paragraph.
  - \hyphenpenalty=50 # the penalty for line breaking at an automatically inserted hyphen
  - \exhyphenpenalty=50 # the penalty for line breaking at an explicit hyphen
  - \binoppenalty=700 # the penalty for breaking a line at a binary operator
  - \relpenalty=500 # the penalty for breaking a line at a relation
  - \clubpenalty=150 # extra penalty for breaking after first line of a paragraph
  - \widowpenalty=150 # extra penalty for breaking before last line of a paragraph
  - \displaywidowpenalty=50 # extra penalty for breaking before last line before a display math
  - \brokenpenalty=100 # extra penalty for page breaking after a hyphenated line
  - \predisplaypenalty=10000 # penalty for breaking before a display
  - \postdisplaypenalty=0 # penalty for breaking after a display
  - \floatingpenalty = 20000 # penalty for splitting an insertion (can only be split footnote in standard LaTeX)
  - \raggedbottom # or \flushbottom
  - \usepackage{float} # keep figures where there are in the text
  - \floatplacement{figure}{H} # keep figures where there are in the text
---

# Цель работы

1. Построить графики изменения числа особей в каждой из трех групп по модели SIR.

2. Рассмотрите, как будет протекать эпидемия в разных случаях.

# Задание

Предположим, что некая популяция, состоящая из N особей, (считаем, что популяция изолирована) подразделяется на три группы. 

- $S(t)$ — восприимчивые к болезни, но пока здоровые особи

- $I(t)$ — это число инфицированных особей, которые также при этом являются распространителями инфекции

- $R(t)$ — это здоровые особи с иммунитетом к болезни. 

До того, как число заболевших не превышает критического значения $I^*$ считаем, что все больные изолированы и не заражают здоровых. Когда $I(t)>I^*$, тогда инфицирование способны заражать восприимчивых к болезни особей.

Таким образом, скорость изменения числа S(t) меняется по следующему закону:

$$ \frac{\partial S}{\partial t} = \begin{cases} - \alpha S, I(t)>I^* \\ 0,  I(t) <= I^* \end{cases}$$

Поскольку каждая восприимчивая к болезни особь, которая, в конце концов, заболевает, сама становится инфекционной, то скорость изменения числа инфекционных особей представляет разность за единицу времени между заразившимися и теми, кто уже болеет и лечится, т.е.:

$$ \frac{\partial I}{\partial t} = \begin{cases} - \alpha S - \beta I, I(t)>I^* \\ - \beta I, I(t) <= I^* \end{cases}$$

А скорость изменения выздоравливающих особей (при этом приобретающие иммунитет к болезни)

$$ \frac{\partial R}{\partial t} = \beta I$$

Постоянные пропорциональности:

- $\alpha$ — коэффициент заболеваемости

- $\beta$ — коэффициент выздоровления

Для того, чтобы решения соответствующих уравнений определялось однозначно, необходимо задать начальные условия. Считаем, что на начало эпидемии в момент времени $t = 0$ нет особей с иммунитетом к болезни $R(0)=0$, а число инфицированных и восприимчивых к болезни особей $I(0)$ и $S(0)$ соответственно. Для анализа картины протекания эпидемии необходимо рассмотреть два случая: $I(0) \leq I^*$ и $I(0) > I^*$



# Выполнение лабораторной работы

$$N=4 578$$
$$I(0)=78$$
$$R(0)=28$$
$$S(0)=N - I(0) - R(0)$$
$$ \alpha = 0.01 $$
$$ \beta = 0.02 $$
Постройте графики изменения числа особей в каждой из трех групп.
Рассмотрите, как будет протекать эпидемия в случае:

- $I(0) \leq I^*$ 

- $I(0) > I^*$

# Код на Python: 

```
a = 0.01 # коэффициент заболеваемости
b = 0.02 # коэффициент выздоровления
N = 4578 # общая численность популяции
I0 = 78 # количество инфицированных особей в начальный момент времени
R0 = 28 #количество здоровых особей с иммунитетом в начальный момент времени
S0 = N - I0 - R0 #количество восприимчивых к болезни особей в начальный момент времени

# случай, когда I(0)<=I*

def syst(x,t):
    dx = 0
    dx1 = -b*x[1]
    dx2 = b*x[1]
    return dx, dx1, dx2

t = np.arange(0, 200, 0.01)

x0 = [S0, I0, R0] # начальные значения

y = odeint(syst, x0, t)

plt.plot(t, y[:,0], label='S(t)')
plt.plot(t, y[:,1], label='I(t)')
plt.plot(t, y[:,2], label='R(t)')
plt.title('I(0)<=I*')
plt.legend()
plt.show()

# случай, когда I(0)>I*

def system(x, t):
    dx_2 = -a*x[0]
    dx_21 = a*x[0] - b*x[1]
    dx_22 = b*x[1]
    return dx_2, dx_21, dx_22

y_2 = odeint(system, x0, t)

plt.plot(t, y_2[:,0], label='S(t)')
plt.plot(t, y_2[:,1], label='I(t)')
plt.plot(t, y_2[:,2], label='R(t)')
plt.title('I(0)>I*')
plt.legend()
plt.show()


```



(рис. fig:001)

![Динамика изменения числа людей в каждой из трех групп в случае, когда $I(0) <= I^*$ с начальными условиями $I(0)=78, R(0)=28, S(0)=4578$.
Коэффициенты $\alpha = 0.01, \beta = 0.02$.](image/Fig_1.jpg){ #fig:001 width=70% }


(рис. fig:002)

![Динамика изменения числа людей в каждой из трех групп в случае, когда $I(0) > I^*$ с начальными условиями $I(0)=78, R(0)=28, S(0)=4578$.
Коэффициенты $\alpha = 0.01, \beta = 0.02$.](image/Fig_2.jpg){ #fig:002 width=70% }





# Выводы


Я ознакомилась с простейшей моделью эпидемии, построив для нее графики изменения числа особей в трех группах для двух случаев: I(0) <= I* и I(0) > I*.

