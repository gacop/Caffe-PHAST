#ifndef __YAML_PARSER_H
#define __YAML_PARSER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <regex>
#include <tuple>

#ifdef DEBUG
#include <iostream>
#endif

namespace utility
{

// tuple utility
namespace detail
{
    template <int... Is>
    struct seq { };

    template <int N, int... Is>
    struct gen_seq : gen_seq<N - 1, N - 1, Is...> { };

    template <int... Is>
    struct gen_seq<0, Is...> : seq<Is...> { };
}

namespace detail
{
	template <typename T>
	void suppress_unused_variable(const T&) {}

    template<typename T, typename F, int... Is>
    void for_each(T&& t, F& f, seq<Is...>)
    {
        auto l = { (f(std::get<Is>(t)), 0)... };
		suppress_unused_variable(l);
    }
    template<typename T, typename F, int... Is>
    void for_each(T&& t, const F& f, seq<Is...>)
    {
        auto l = { (f(std::get<Is>(t)), 0)... };
		suppress_unused_variable(l);
    }
}

template <typename... Ts, typename F>
void for_each(std::tuple<Ts...>& t, F& f)
{
    detail::for_each(t, f, detail::gen_seq<sizeof...(Ts)>());
}

template <typename... Ts, typename F>
void for_each(std::tuple<Ts...>& t, const F& f)
{
    detail::for_each(t, f, detail::gen_seq<sizeof...(Ts)>());
}

template <typename... Ts, typename F>
void for_each(const std::tuple<Ts...>& t, F& f)
{
    detail::for_each(t, f, detail::gen_seq<sizeof...(Ts)>());
}

template <typename... Ts, typename F>
void for_each(const std::tuple<Ts...>& t, const F& f)
{
    detail::for_each(t, f, detail::gen_seq<sizeof...(Ts)>());
}

template <typename Value>
struct comparer
{
	comparer(const Value& value) : value_(value), p_output_(nullptr) {}

	template <typename TupleValue>
	void operator()(const TupleValue& p)
	{
		if(!p_output_ && p == value_)
			p_output_ = &p;
	}

	const Value& value_;
	const Value* p_output_;
};

template <typename... Ts, typename Value>
const Value* find(const std::tuple<Ts...>& t, const Value& value)
{
	comparer<Value> comp(value);
	for_each(t, comp);
	return comp.p_output_;
}

// string utility
inline bool starts_with(const std::string& str, const std::string& sub)
{
	return str.rfind(sub, 0) != std::string::npos;
}
inline bool ends_with(const std::string& str, const std::string& sub)
{
	return str.rfind(sub, str.length()-1) != std::string::npos;
}

// param info utility
template <typename T>
struct param_info
{
	param_info(std::string name, int offset) : name_(name), offset_(offset) {}

	using param_type = T;

	std::string name_;
	int offset_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const param_info<T>& p)
{
    os << p.name_ << " - " << p.offset_;
    return os;
}

template <typename T, typename U>
void operator==(const param_info<T>& p1, const param_info<U>& p2)
{
	return p1.name_ == p2.name_;
}

struct printer
{
	template <typename T>
	void operator()(const param_info<T>& p) const
	{
		std::cout << p << std::endl;
	}
};

template <typename ParamsStruct>
struct printer_with_value
{
	printer_with_value(ParamsStruct params) : params_(params) {}

	template <typename T>
	void operator()(const param_info<T>& p) const
	{
		std::cout << p.name_ << ": " << ((T*)(p.offset_ + (char*)(&params_)))[0] << ", ";
	}

	ParamsStruct params_;
};

struct indent_adder
{
	indent_adder(std::string indent) : indent_(indent) {}

	template <typename T>
	void operator()(param_info<T>& p)
	{
		p.name_ = indent_ + p.name_ + ":";
	}

	std::string indent_;
};

template <typename ParamsStruct>
struct populator
{
	populator(ParamsStruct& params, std::string row, std::string indent) :
		params_(params), row_(row), indent_(indent), good_(false) {}

	template <typename T>
	void operator()(const param_info<T>& info)
	{
		if(!starts_with(row_, indent_))
		{
			std::stringstream ss;
			ss << "wrong indentation at line: " << row_;
			throw std::runtime_error(ss.str());
		}

		std::string indented_param(indent_ + info.name_);
		if(row_.find(indented_param) != 0)
			return;

		int pos = (indented_param + std::string(": ")).length();
		std::string value_string = row_.substr(pos, row_.length() + (ends_with(row_, "\n") ? (-1) : (0)) - pos);

		typename param_info<T>::param_type value;
		try
		{
			std::stringstream ss;
			ss << value_string;
			ss >> value;
		}
		catch(...)
		{
			std::stringstream ss;
			ss << "error converting integer value at line: " << row_;
			throw std::runtime_error(ss.str());
		}

		typename param_info<T>::param_type* out_address = (typename param_info<T>::param_type*)((char*)(&params_) + info.offset_);
		*out_address = value;

		good_ = true;
	}

	bool good() const
	{
		return good_;
	}

	ParamsStruct& params_;
	std::string row_;
	std::string indent_;
	bool good_;
};

// parser
template <typename ParamsStruct>
class yaml_parser
{
public:
	template <typename... T>
	yaml_parser(const std::string& str, std::tuple<param_info<T>...> param_traits) : algos_()
	{
		std::string str_(str);
		str_ = std::regex_replace(str_, std::regex("[ \\t]+:"), ":");
		str_ = std::regex_replace(str_, std::regex(":[ \\t]+"), ": ");
		str_ = std::regex_replace(str_, std::regex("\\r\\n"), "\n");
		str_ = std::regex_replace(str_, std::regex("[ \\t]+\\n"), "\n");
		str_ = std::regex_replace(str_, std::regex("#.*"), "");

		auto rows = get_rows(str_);
#if DEBUG
		std::cout << "String to process:" << std::endl;
		for(const std::string& row : rows)
			std::cout << row << std::endl;

		std::cout << "Param info:" << std::endl;
		for_each(param_traits, printer());
#endif

		std::vector<int> start_lines;
		for(int i = 0; i < (int)rows.size(); ++i)
		{
			if(starts_with(rows[i], "- ") && ends_with(rows[i], ":"))
				start_lines.push_back(i);
		}
		if(start_lines.empty())
		{
			//throw std::runtime_error("no YAML dictionary found");
			return;
		}

		std::string indent;
		for(int i = 0; i < (int)start_lines.size(); ++i)
		{
			int index = start_lines[i];
			std::string	name = rows[index].substr(2, rows[index].length() - 3);
#if DEBUG
			std::cout << "found dictionary entry " << name << std::endl;
#endif

			int begin_content = index + 1;
			int end_content = (i == (int)start_lines.size() - 1) ? rows.size() : start_lines[i + 1];

			if(begin_content == end_content) // empty dictionary
			{
				algos_[name] = ParamsStruct();
				continue;
			}

			if(indent.empty())
			{
				if(starts_with(rows[index+1], "    "))
					indent = "    ";
				else if(starts_with(rows[index+1], "  "))
					indent = "  ";
				else
					std::runtime_error("wrong indentation");
			}

			// get content & populate ParamsStruct
			ParamsStruct params;
			for(int j = begin_content; j < end_content; ++j)
			{
				populator<ParamsStruct> populator(params, rows[j], indent);
				for_each(param_traits, populator);
				if(!populator.good())
				{
					std::stringstream ss;
					ss << "problematic line: " << rows[j];
					throw std::runtime_error(ss.str());
				}
			}

			algos_[name] = params;
		}
#if DEBUG
		for(auto pair : algos_)
		{
			std::cout << pair.first << ": ";
			for_each(param_traits, printer_with_value<ParamsStruct>(pair.second));
			std::cout << std::endl;
		}
#endif
	}

	const ParamsStruct& get_elem(std::string key) const
	{
		return algos_.at(key);
	}

private:
	std::unordered_map<std::string, ParamsStruct> algos_;

	std::vector<std::string> get_rows(const std::string& str)
	{
		std::vector<std::string> rows;
		const char* c_str = str.c_str();
		const char* end_str = c_str + str.length();
		for(const char* c = c_str; c <= end_str; ++c)
		{
			if(*c == '\n' || c == end_str)
			{
				if(c != c_str)
					rows.emplace_back(c_str, c - c_str);
				c_str = c + 1;
			}
		}
		return rows;
	}
};

template <typename Class, typename Member>
size_t offset(Member Class::*member)
{
	return (const char*)&(((Class*)nullptr)->*member) - (const char*)nullptr;
} 

}

#endif /* __YAML_PARSER_H */
