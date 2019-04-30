/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#include "Output.h"

namespace SHOT
{
Output::Output()
{
    consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("multisink.txt", true);
    std::vector<spdlog::sink_ptr> sinks{ consoleSink, fileSink };
    logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());

    logger->set_pattern("%v");

    // The maximum level of logging to use by all sinks
    logger->set_level(spdlog::level::trace);

    setLogLevels(E_LogLevel::Info, E_LogLevel::Info);
}

Output::~Output() = default;

void Output::outputCritical(std::string message) { logger->critical(message); }

void Output::outputError(std::string message) { logger->error(message); }

void Output::outputError(std::string message, std::string errormessage)
{
    logger->error("{}: \"{}\"", message, errormessage);
}

void Output::outputWarning(std::string message) { logger->warn(message); }

void Output::outputInfo(std::string message) { logger->info(message); }

void Output::outputDebug(std::string message) { logger->debug(message); }

void Output::outputTrace(std::string message)
{
#ifndef NDEBUG
    logger->trace(message);
#endif
}

void Output::setLogLevels(E_LogLevel consoleLogLevel, E_LogLevel fileLogLevel)
{
    // Sets the correct log levels

    switch(consoleLogLevel)
    {
    case E_LogLevel::Off:
        consoleSink->set_level(spdlog::level::off);
        break;

    case E_LogLevel::Critical:
        consoleSink->set_level(spdlog::level::critical);
        break;

    case E_LogLevel::Error:
        consoleSink->set_level(spdlog::level::err);
        break;

    case E_LogLevel::Warning:
        consoleSink->set_level(spdlog::level::warn);
        break;

    case E_LogLevel::Info:
        consoleSink->set_level(spdlog::level::info);
        break;

    case E_LogLevel::Debug:
        consoleSink->set_level(spdlog::level::debug);
        break;

    case E_LogLevel::Trace:
        consoleSink->set_level(spdlog::level::trace);
        break;

    default:
        break;
    }

    switch(fileLogLevel)
    {
    case E_LogLevel::Off:
        fileSink->set_level(spdlog::level::off);
        break;

    case E_LogLevel::Critical:
        fileSink->set_level(spdlog::level::critical);
        break;

    case E_LogLevel::Error:
        fileSink->set_level(spdlog::level::err);
        break;

    case E_LogLevel::Warning:
        fileSink->set_level(spdlog::level::warn);
        break;

    case E_LogLevel::Info:
        fileSink->set_level(spdlog::level::info);
        break;

    case E_LogLevel::Debug:
        fileSink->set_level(spdlog::level::debug);
        break;

    case E_LogLevel::Trace:
        fileSink->set_level(spdlog::level::trace);
        break;

    default:
        break;
    }

    // Also set the level for the main logger
    if((int)consoleLogLevel <= (int)fileLogLevel)
    {
        logger->set_level((spdlog::level::level_enum)consoleLogLevel);
    }
    else
    {
        logger->set_level((spdlog::level::level_enum)fileLogLevel);
    }
}

void Output::setConsoleSink(std::shared_ptr<spdlog::sinks::sink> newSink)
{
    // copy loglevel from previous consoleSink
    newSink->set_level(consoleSink->level());
    // set our pattern
    newSink->set_pattern("%v");

    // install new consoleSink
    consoleSink = newSink;
    logger->sinks()[0] = consoleSink;
}

} // namespace SHOT