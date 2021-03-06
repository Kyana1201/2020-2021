---
lang: ru-RU
title: Лабораторная работа №4
author: |
    Khokhlacheva Yana Dmitrievna\inst{1}
institute: |
    \inst{1}RUDN University, Moscow, Russian Federation
date: RUDN University, 06 March, 2021 Moscow, Russia


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
Понимание приципов построения модели линейного гармонического осциллятора позволяет описывать движение грузика на пружинке, маятника, заряда в электрическом контуре, а также эволюцию во времени многих систем в физике, химии, биологии и других науках с помощью одного и того же уравнения.  

## Цель работы
Рассмотреть модель гармонических колебаний

## Задачи
Построите фазовый портрет гармонического осциллятора и решение уравнения
гармонического осциллятора для следующих случаев
1. Колебания гармонического осциллятора без затуханий и без действий внешней силы $\ddot {x} + 4.3x = 0$
2. Колебания гармонического осциллятора c затуханием и без действий внешней силы $\ddot {x} + 6 \dot {x} + 5x = 0$
3. Колебания гармонического осциллятора c затуханием и под действием внешней силы $\ddot {x} + 10 \dot {x} + 9x = 8sin(7t)$

На интервале $t \in [0; 80]$(шаг 0.05) с начальными условиями $x_0 = 0.8, y_0 = -1.2$


## Результат
В данной лабораторной работе ознакомилася с моделью линейного гармонического осциллятора, решив уравнения гармонического осциллятора и построив его фазовые портреты. 

(рис. fig:001)

![Колебания гармонического осциллятора без затуханий и без действий внешней
силы](image/Fig1.jpg){ #fig:001 width=70% }


(рис. fig:002)

![Колебания гармонического осциллятора c затуханием и без действий внешней
силы](image/Fig2.jpg){ #fig:002 width=70% }


(рис. -@fig:003)

![Колебания гармонического осциллятора c затуханием и под действием внешней
силы](image/Fig3.jpg){ #fig:003 width=70% }


