#ifndef __CONFIGURATION_FILE_H
#define __CONFIGURATION_FILE_H

#if __cplusplus > 201703L
#include <filesystem>
#endif
#include <fstream>
#include <streambuf>
#include "yaml_parser.h"

namespace phast
{
	class configuration_file
	{
		struct params
		{
			params() : major_block_size(-1), minor_block_size(-1), scheduling_strategy((int)phast::scheduling_strategy::QUEUE),
				tiling_strategy((int)phast::tiling_strategy::N_ELEM_PER_BLOCK), shared_pre_load(0),
				n_thread(phast::custom::multi_core::get_max_threads()), w_package(1), w_core(100), w_hyperthread(10000)
			{}

			int major_block_size;
			int minor_block_size;
			int scheduling_strategy;
			int tiling_strategy;
			int shared_pre_load;
			int n_thread;
			int w_package;
			int w_core;
			int w_hyperthread;
		};

	public:
		static configuration_file& get_instance(std::string name = std::string())
		{
			static configuration_file instance(name);
			return instance;
		}

		static void retrieve_parameters(const std::string& key)
		{
			get_instance()._retrieve_parameters(key);
		}

		void _retrieve_parameters(const std::string& key) const
		{
			try
			{
				auto params = parser_->get_elem(key);
				phast::custom::cuda::set_block_size(params.major_block_size, params.minor_block_size);
				phast::custom::cuda::set_scheduling_strategy((phast::scheduling_strategy)params.scheduling_strategy);
				phast::custom::cuda::set_tiling_strategy((phast::tiling_strategy)params.tiling_strategy);
				phast::custom::cuda::set_shared_pre_load(params.shared_pre_load);
				phast::custom::multi_core::set_n_thread(params.n_thread);
				phast::custom::multi_core::set_affinity_weights(params.w_package, params.w_core, params.w_hyperthread);
			}
			catch(...)
			{
				// NOP, keep previous settings
#if DEBUG
				std::cerr << "Cannot find parameters for " << key << std::endl;
#endif
			}
		}

	private:
		configuration_file(std::string name)
		{
			if(name.empty())
				throw std::runtime_error("Please, provide a valid filename the first time get_instance() is invoked");

			name_ = name;
#if __cplusplus > 201703L
			if(!std::filesystem::exists(name_) || !std::filesystem::is_regular_file(name_))
			{
				std::stringstream ss;
				ss << name_ << " is not a valid configuration file";
				throw std::runtime_error(ss.str());
			}
#endif
			std::ifstream param_file(name_);
			if(!param_file)
			{
				std::stringstream ss;
				ss << name_ << " is not a valid configuration file";
				throw std::runtime_error(ss.str());
			}

			std::string str((std::istreambuf_iterator<char>(param_file)), std::istreambuf_iterator<char>());
			parser_.reset(new utility::yaml_parser<params>(str, std::make_tuple(
				utility::param_info<int>("major_block_size", utility::offset(&params::major_block_size)),
				utility::param_info<int>("minor_block_size", utility::offset(&params::minor_block_size)),
				utility::param_info<int>("scheduling_strategy", utility::offset(&params::scheduling_strategy)),
				utility::param_info<int>("tiling_strategy", utility::offset(&params::tiling_strategy)),
				utility::param_info<int>("shared_pre_load", utility::offset(&params::shared_pre_load)),
				utility::param_info<int>("n_thread", utility::offset(&params::n_thread)),
				utility::param_info<int>("w_package", utility::offset(&params::w_package)),
				utility::param_info<int>("w_core", utility::offset(&params::w_core)),
				utility::param_info<int>("w_hyperthread", utility::offset(&params::w_hyperthread))
			)));
		}

		std::string name_;
		std::unique_ptr<utility::yaml_parser<params>> parser_;
	};
}


#endif /* __CONFIGURATION_FILE_H */
