---
layout: post
title: Automatic differentiation with nilpotent numbers
tags: Programming Math C++
---

When first learning about derivatives in calculus, we often see a definition along the lines of

$$
    f'(x) \equiv \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.
$$

This definition works just fine, and gives a good intuition for what a derivative is. However, it isn't a definition can be readily implemented in some programmatic way. The most obvious way to do this is through forward finite differences: we choose some small value for $$ h $$, and then compute the above expression (without the limit) directly. The quality of the approximation will depend on the function and on the value of $$ h $$ that is chosen. <!-- read more --> Naively, you might expect that you can pick some arbitrarily small number and have it work just fine, and you would be absolutely correct if it weren't for the computational limitations of floating point numbers. Unfortunately, things like [round-off error](https://en.wikipedia.org/wiki/Round-off_error) and [catastrophic cancellation](https://en.wikipedia.org/wiki/Loss_of_significance) make it impossible to find a suitable choice of $$ h $$ that works for any function.

Here's a simple example showing how this breaks down. Let $$ [x] $$ represent the floating point 
approximation of the real value $$ x $$, and consider the function $$ f(x) = x $$. We will
choose $$ h = 2^{-53} $$; then

$$
    f'(x)
    \approx \frac{f(x + h) - f(x)}{h}
    = \frac{(x + 2^{-53}) - x}{2^{-53}}
    = 1.
$$

This happens to be exact for any value of $$ x $$ in this case, and in fact we'd get the same result no matter what choice of $$ h $$ we make due to the fact that we chose a linear function. But this doesn't work on a typical computer! Assuming that we use IEEE 754 compliant 64-bit floating point numbers (as is usually the case for the `double` type in C/C++, or the `float` type in Python), we'd find that our result depends on $$ x $$. For example,

$$
    [f'(0.0)] = 1.0
    \quad \text{and} \quad
    [f'(1.0)] = 0.0
$$

The problem is our poor choice of $$ h $$. It turns out that $$ [1.0 + 2^{-53}] = [1.0] $$, leading to the unfortunate result that $$ [f(x + h)] - [f(x)] = [0.0] $$. Not only that, even if our choice of $$ h $$ is "good" for for our use case, the result is still just an approximation. 

Let's see if we can do better.

## Dual numbers

The dual numbers over the reals $$ \mathbb{R} $$ can be defined as the set of ordered pairs $$ (a, b) \in \mathbb{R}^2 $$ with addition and multiplication given by

$$
    (a, b) + (c, d) = (a + c, b + d)
    \quad \text{and} \quad
    (a, b) \times (c, d) = (ac, ad + bc).
$$

for all $$ (a, b), (c, d) \in \mathbb{R}^2 $$.

The reason for these particular rules comes from considering an infinitesimal value $$ \epsilon $$ that is nilpotent, so that $$ \epsilon^2 = 0 $$. If we extend the reals by considering values of the form $$ a + b \epsilon $$, then we see that

$$
    (a + b\epsilon) + (c + d\epsilon) = (a + c) + (b + d)\epsilon
$$

and

$$
    (a + b\epsilon)(c + d\epsilon)
    = ac + (ad + bc) \epsilon + bd \epsilon^2
    = ac + (ad + bc) \epsilon,
$$

for all $$ a, b, c, d \in \mathbb{R} $$. We can also obtain a division rule:

$$
    \frac{a + b\epsilon}{c + d\epsilon}
    = \frac{a + b\epsilon}{c + d\epsilon} \times \frac{c - d\epsilon}{c - d\epsilon}
    = \frac{a c + (bc - ad) \epsilon}{c^2}
    = \frac{a}{c} + \left(\frac{bc - ad}{c^2}\right)\epsilon
$$

which is defined provided $$ c \neq 0 $$. These rules probably seem familiar due to similarities to the complex numbers.

Another way to define the dual numbers is via matrices. Let

$$
    I = \begin{bmatrix}
        1 & 0 \\
        0 & 1
    \end{bmatrix}
    \quad \text{and} \quad
    \epsilon = \begin{bmatrix}
        0 & 1 \\
        0 & 0
    \end{bmatrix}.
$$

Then we can identify the dual number $$ (a, b) $$ with the matrix

$$
    \begin{bmatrix}
        a & b \\
        0 & a
    \end{bmatrix}
    = aI + b\epsilon.
$$

It's easy to check that normal matrix addition and multiplication reproduce the rules we set above.

The benefit of dual numbers is that we obtain derivatives automatically. For example, consider some function $$ f(x) $$. Expanding $$ f(a + b\epsilon) $$ in a Taylor series about $$ a $$, we have that

$$
    f(a + b\epsilon) = f(a) + f'(a) b \epsilon + \frac{1}{2} f''(a) (b \epsilon)^2 + \cdots
$$

Using the fact that $$ \epsilon^n = 0 $$ for $$ n \geq 2 $$, this becomes a relation

$$
    f(a + b\epsilon) = f(a) + b f'(a) \epsilon.
$$

Thus, for some function $$ f(x) $$ defined on the reals, we can extend it to work on the dual numbers using this rule. It's particularly useful to consider dual numbers of the form $$ \tilde{x} = x + \epsilon $$, since then

$$
    f(\tilde{x}) = f(x) + f'(x) \epsilon.
$$

We can see this in action by looking at a simple function like $$ f(x) = x^2 $$. Then

$$
    f(x + \epsilon) = (x + \epsilon)^2 = x^2 + 2x \epsilon.
$$

This can be further extended to work for directional derivatives of multivariable functions. For example, consider the function $$ f(x, y) = x^2 y $$. Let's suppose we want to compute the derivative in the direction $$ \vec{p} = u \hat{x} + v \hat{y} $$. We can see that 

$$
    f(x + u\epsilon, y + v\epsilon)
    = (x + u\epsilon)^2 (y + v \epsilon)
    = (x^2 + 2xu \epsilon) (y + v\epsilon)
    = x^2 y + (2xyu + x^2v) \epsilon
$$

from which we can identify the directional derivative

$$
    \nabla f \cdot \vec{p} = 2xyu + x^2v.
$$

## Automatic differentiation

Dual numbers are pretty simple to implement. Consider the following:

```c++
class dual {
public:
    constexpr dual(double real) : dual(real, T{}) {}
    constexpr dual(double real, double diff) : m_data{real, diff} {}

    constexpr auto real() { return m_data[0]; }
    constexpr auto diff() { return m_data[1]; }

    dual& operator+=(dual rhs) noexcept {
        m_data[0] += rhs.m_data[0];
        m_data[1] += rhs.m_data[1];
        return *this;
    }

    dual& operator-=(dual rhs) noexcept {
        m_data[0] -= rhs.m_data[0];
        m_data[1] -= rhs.m_data[1];
        return *this;
    }

    dual& operator*=(dual rhs) noexcept {
        auto [a, b] = m_data;
        auto [c, d] = rhs.m_data;
        m_data[0] = a * c;
        m_data[1] = a * d + b * c;
        return *this;
    }

    dual& operator/=(dual rhs) noexcept {
        auto [a, b] = m_data;
        auto [c, d] = rhs.m_data;
        m_data[0] = a / c;
        m_data[1] = (b * c - a * d) / (c * c);
        return *this;
    }

private:
    double m_data[2];
};

auto operator+(dual lhs, dual rhs) noexcept { return lhs += rhs; }
auto operator-(dual lhs, dual rhs) noexcept { return lhs -= rhs; }
auto operator*(dual lhs, dual rhs) noexcept { return lhs *= rhs; }
auto operator/(dual lhs, dual rhs) noexcept { return lhs /= rhs; }
```

This defines a `dual` class that implements the basic arithmetic operations. Now, we can do things
like

```c++
dual x = {0.5, 1};
auto y = 1.0 / (1.0 - x);

std::cout << y.real() << '\n'; // prints "2.0"
std::cout << y.diff() << '\n'; // prints "4.0"
```

Great! With no additional work on our part (other than making `x` a `dual` instead of a 
normal floating point type), we obtained both the value and the derivative of the function 
$$ f(x) = 1 / (1 - x) $$ at $$ x = 1/2 $$. 

To handle more complex functions, we need only specialize the standard library math functions for
our `dual` type. As an example, we can specialize `std::sin`

```c++
namespace std {

auto sin(dual x) -> dual {
    auto a = sin(x.real());
    auto b = cos(x.real());
    return {a, b * x.diff()};
}

}  // namespace std
```

which allows us to evaluate more complex functions, such as

```c++
auto x = dual{0.25, 1.0};
auto y = std::sin(1.0 / (1.0 - x));

std::cout << y.real() << '\n'; // prints "0.971938"
std::cout << y.diff() << '\n'; // prints "0.418200"
```

Besides the fact that we get the value of the derivative automatically, we also avoided the problem
inherent in a finite difference approximation of the derivative, which is the choice of the value of
$$ h $$. Rather than needing to make this choice on a per-function basis, the calculation is all 
done behind the scenes, with no approximations. It just works&trade;.


