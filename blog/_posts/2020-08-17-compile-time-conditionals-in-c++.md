---
title: Compile time conditionals in C++
layout: post
tags: Programming C++
---

Often when doing template metaprogramming in C++, we run into the issue of dealing with compile-time conditionals. Though we can use `std::conditional` as a compile-time equivalent to `if`, there isn't a compile-time equivalent to `switch` in the standard library which, in my opinion, could lead to more succinct and readable code. <!-- read more -->

For example, I was recently looking into random number generators, and came across [a paper by Daniel Lemire](https://dl.acm.org/doi/10.1145/3230636) detailing an algorithm to produce unbiased integers in a given interval. The specifics of the algorithm aren't the focus here, except for one key point: it requires an integral type with twice as many bits as the desired output type in order to store full result of a multiplication. This is a pretty straightforward problem to solve using a chain of `std::conditional`s:

```c++
using output_t = /* unsigned integral type with desired number of bits */
using bigint_t = std::conditional_t<
        /* if   */ (sizeof(output_t) == 1),
        /* then */ std::uint16_t,
        /* else */ std::conditional_t<
            /* if   */ (sizeof(output_t) == 2),
            /* then */ std::uint32_t,
            /* else */ std::conditional_t<
                /* if   */ (sizeof(output_t) == 4),
                /* then */ std::uint64_t,
                /* else */ __uint128_t // <-- if your architecture supports it
                >
            >
        >;
```

It certainly works, but it's not the prettiest, and can get nasty if the conditionals nest deep. For cases like this, it would be really nice if we had some kind of compile-time equivalent to the switch statement.

We'll take a page from Boost's book and define compile-time equivalents to keywords by appending an underscore. So in this case, we'll define `switch_`, `case_`, and `default_`. The implementation of `case_` is straightforward:

```c++
template <auto Value, class Type>
struct case_ {
    using type = Type;
    static constexpr auto value = Value;
};
```

We'll implement `default_` as a value of type `std::nullptr_t`. This works since `nullptr` is allowed as a non-type template parameter, though it does introduce some limitations if you ever want to actually match on `nullptr` [^1].

<!-- markdownlint-disable code-block-style -->
[^1]:
    In C++20 this is no longer an issue, since non-type template parameters with class type are allowed, so we can instead define `default_` as

    ```c++
    struct default_tag {};
    constexpr auto default_ = default_tag{};
    ```
    
    which won't conflict with any other type.
<!-- markdownlint-enable code-block-style -->

```c++
constexpr auto default_ = nullptr;
```

The switch needs a condition value and a list of cases. We recursively check each case until either we find a case whose value matches the condition value, or whose value is `default_`. If no case is found, we can provide a nice error message.

```c++
namespace detail {
    template <auto Condition, auto Value>
    struct check_case {
        using cond_t = decltype(Condition);
        using value_t = decltype(Value);
        static constexpr auto value =
            std::is_same_v<value_t, decltype(default_)> || (Condition == Value);
    };
}  // namespace detail

template <auto Condition, class First, class... Rest>
struct switch_ {
    using type = typename std::conditional_t<
        /* if   */ detail::check_case<Condition, First::value>::value,
        /* then */ First,
        /* else */ switch_<Condition, Rest...>
    >::type;
};

// base case
template <auto Condition, class Case>
struct switch_<Condition, Case> {
    static_assert(detail::check_case<Condition, Case::value>::value,
                  "Error: switch_ failed as no case matches the condition value");
    using type = typename Case::type;
};

template <auto Condition, class... Cases>
using switch_t = typename switch_<Condition, Cases...>::type;
```

Now we can use it!

```c++
using output_t = /* unsigned integral type with desired number of bits */
using bigint_t = switch_t<sizeof(output_t),
            case_<1, std::uint16_t>,
            case_<2, std::uint32_t>,
            case_<4, std::uint64_t>,
            case_<8, __uint128_t>,
            case_<default_, void>
        >;
```

This looks much nicer to me than the original nested `std::conditional`s.
