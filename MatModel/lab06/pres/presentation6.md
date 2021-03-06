---
lang: ru-RU
title: Лабораторная работа №6
author: |
    Khokhlacheva Yana Dmitrievna\inst{1}
institute: |
    \inst{1}RUDN University, Moscow, Russian Federation
date: RUDN University, 27 February, 2021 Moscow, Russia


## Formatting
toc: false
slide_level: 2
theme: metropolis
header-includes: 
 - \metroset{progressbar=frametitle,sectionpage=progressbar,numbering=fraction}
 - '\makeatletter'
 - '\beamer@ignorenonframefalse'
 - '\makeatother'
aspectratio: 43
section-titles: true
---

## Прагматика выполнения лабораторной работы(Зачем)

Понимание принципов построения модели "Эпидемия" и знание соответствующих модельных уравнений позволяет выявить тенденции к росту/падению числа особей, восприимчивых к болезни, инфицированных особей, а также здоровых особей.

## Цель работы

• Ознакомиться с простейшей моделью Эпидемии

• Некая популяция, состоящая из N особей, подразделяется на три группы:

1. Восприимчивые к болезни, но пока здоровые особи S.

2. Инфицированных особей, которые также при этом являются распространителями инфекции I.

3. Здоровые особи с иммунитетом к болезни R.

## Задачи

1. Построить графики изменения числа особей в каждой из трех групп (восприимчивые к болезни (S), заболевшие люди (I), здоровые люди с иммунитетом (R)), если I(0) \leq I* (число инфицированных не превышает критического значения).

2. Построить графики изменения числа особей в каждой из трех групп (восприимчивые к болезни (S), заболевшие люди (I), здоровые люди с иммунитетом (R)), если I(0) > I* (число инфицированных выше критического значения).

## Результат

В данной лабораторной работе рассмотрела простейшие модели эпидемии, а также научилась строить динамику изменения числа особей в каждой из трех групп (восприимчивые к болезни (S), заболевшие люди (I), здоровые люди с иммунитетом (R)) для двух случаев: I(0) <= I* и I(0) > I*

(рис. -@fig:001)

![Динамика изменения числа людей в каждой из трех групп в случае, когда $I(0) <= I^*$ с начальными условиями $I(0)=78, R(0)=28, S(0)=4578$.
Коэффициенты $\alpha = 0.01, \beta = 0.02$](image/Fig_1.jpg){ #fig:001 width=70% }


(рис. -@fig:002)

![Динамика изменения числа людей в каждой из трех групп в случае, когда $I(0) > I^*$ с начальными условиями $I(0)=78, R(0)=28, S(0)=4578$.
Коэффициенты $\alpha = 0.01, \beta = 0.02$.](image/Fig_2.pjpg){ #fig:002 width=70% }



