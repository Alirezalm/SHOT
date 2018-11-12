/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#include "NonlinearExpressions.h"

namespace SHOT
{
std::ostream &operator<<(std::ostream &stream, NonlinearExpressionPtr expr)
{
    stream << *expr;
    return stream;
};
} // namespace SHOT