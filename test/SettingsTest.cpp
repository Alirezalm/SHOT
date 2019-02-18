/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#include "Shared.h"
#include "Settings.h"
#include "SHOTSolver.h"

using namespace SHOT;

bool SettingsTestOptions(bool useOSiL);

int SettingsTest(int argc, char* argv[])
{
    int defaultchoice = 1;

    int choice = defaultchoice;

    if(argc > 1)
    {
        if(sscanf(argv[1], "%d", &choice) != 1)
        {
            printf("Couldn't parse that input as a number\n");
            return -1;
        }
    }

    bool passed = true;

    switch(choice)
    {
    case 1:
        std::cout << "Starting test to read and write OSoL files:" << std::endl;
        passed = SettingsTestOptions(true);
        std::cout << "Finished test to read and write OSoL files." << std::endl;
        break;
    case 2:
        std::cout << "Starting test to read and write opt files:" << std::endl;
        passed = SettingsTestOptions(false);
        std::cout << "Finished test to read and write opt files." << std::endl;
        break;
    default:
        passed = false;
        std::cout << "Test #" << choice << " does not exist!\n";
        break;
    }

    if(passed)
        return 0;
    else
        return -1;
}

// Test the writing and reading of options files
bool SettingsTestOptions(bool useOSiL)
{
    bool passed = true;

    std::unique_ptr<SHOTSolver> solver = std::make_unique<SHOTSolver>();

    std::string filename;

    if(useOSiL)
    {
        filename = "options.xml";
    }
    else
    {
        filename = "options.opt";
    }

    if(boost::filesystem::exists(filename))
        std::remove(filename.c_str());

    try
    {
        if(useOSiL)
        {
            if(!UtilityFunctions::writeStringToFile(filename, solver->getOptionsOSoL()))
            {
                passed = false;
            }
        }
        else
        {
            if(!UtilityFunctions::writeStringToFile(filename, solver->getOptions()))
            {
                passed = false;
            }
        }

        if(passed && !solver->setOptions(filename))
        {
            std::cout << "Could not read OSoL file." << std::endl;
            passed = false;
        }
    }
    catch(ErrorClass& e)
    {
        std::cout << "Error: " << e.errormsg << std::endl;
        passed = false;
    }

    return passed;
}