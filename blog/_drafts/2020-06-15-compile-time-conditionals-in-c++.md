---
title: Compile time conditionals in C++
layout: post
tags: Programming C++
---



```c++
template <bool Cond, class IfTrue, class IfFalse>
struct eval_if {
    using type = typename std::conditional_t<Cond, IfTrue, IfFalse>::type;
}

template <bool Cond, class IfTrue, class IfFalse>
using eval_if_t = typename eval_if<Cond, IfTrue, IfFalse>::type;
```

```c++
template <class T>
struct identity {
    using type = T;
};
```

```c++
template <bool Cond, class IfTrue>
struct when {
    using type = eval_if_t<Cond, IfTrue, identity<void>>;
    static constexpr bool value = Cond;
};
```

```c++
template <class Case, class... Rest>
struct select {
    using type = eval_if_t<Case::value, Case::type, select<Rest...>>;
};

template <class Case>
struct select<Case> {
    static_assert(Cond::value);
    using type = typename Case::type;
};

template <class... Cases>
using select_t = typename select<Cases...>::type;
```

```c++
template <size_t Bits>
struct int_least_bits {
    using type = select_t<
        when<(Bits <=  8), std::int8_t>,
        when<(Bits <= 16), std::int16_t>,
        when<(Bits <= 32), std::int32_t>,
        when<(Bits <= 64), std::int64_t>
    >;
};

template <size_t Bits>
using int_least_bits_t = typename int_least_bits<Bits>::type;

static_assert(std::is_same_v<std::int8_t, int_least_bits_t<5>>);
```

```c++
template <auto Value, class T>
struct case_ {
    using type = T;
    static constexpr auto value = Value;
};
```

```c++
template <auto Value, class Case, class... Rest>
struct switch_ {
    using type = eval_if_t<(Value == Case::value), Case::type, switch_<Value, Rest...>>;
};

template <auto Value, class... Cases>
using switch_t = typename switch_<Value, Cases...>::type;
```

```c++
template <size_t Bits>
struct int_least_bits {
    using type = switch_<Bits,
        case_< 8, std::int8_t>,
        case_<16, std::int16_t>,
        case_<32, std::int32_t>,
        case_<64, std::int64_t>
    >;
};

template <size_t Bits>
using int_least_bits_t = typename int_least_bits<Bits>::type;

static_assert(std::is_same_v<std::int8_t, int_least_bits_t<5>>);
```